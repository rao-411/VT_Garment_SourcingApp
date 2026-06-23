# cost_predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== Global Variables & Initialization ====================
print("Loading cost predictor...")

# Load pre-trained models with graceful error handling
try:
    models = joblib.load('models.pkl')
    model2 = joblib.load('model2.pkl')
    
    with open('incoterm_rules.json', 'r', encoding='utf-8') as f:
        INCOTERM_RULES = json.load(f)
    
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        TARGETS = metadata.get('targets', ['Exwork(M)', 'Freight(O)', 'Local(Q)', 'Brokerage(S)'])
        
except Exception as e:
    print(f"⚠️  Model loading failed: {e}")
    print("Make sure models.pkl, model2.pkl, incoterm_rules.json, and model_metadata.pkl are in the same directory.")
    models = {}
    model2 = None
    INCOTERM_RULES = {}
    TARGETS = []

# System boundary options
SHIP_VIA_OPTIONS = ['SEA', 'AIR', 'DHL', 'FED']

print("✅ Cost predictor components loaded successfully!\n")


# ==================== Data Preprocessing Pipeline ====================
def data_preprocessing(df):
    """
    Transforms raw Syteline ERP manifests into uniform features.
    """
    if df is None or df.empty:
        raise ValueError("Please provide valid input data")

    df_result = df.copy()
    df_result.columns = df_result.columns.str.strip()

    # Isolate general transit mode profile
    df_result["Ship Via Category"] = df_result["Ship From"].apply(
        lambda x: "AIR" if "BY AIR" in str(x).upper() else "SHIP"
    )

    # Categorize destination
    df_result["destination"] = df_result["Ship From"].apply(
        lambda x: "MM" if "TO MM" in str(x).upper() else "Thailand"
    )

    # Clean Ship From column
    df_result["Ship From"] = (
        df_result["Ship From"]
        .str.replace("BY AIR", "", case=False, regex=False)
        .str.replace("TO MM", "", case=False, regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    return df_result


# ==================== Demand Consolidation Engine ====================
def merge_duplicate_rows(df):
    """
    Consolidates split SKU requests within a 14-day window for the same lane.
    """
    if df is None or df.empty:
        return df

    df_merged = df.copy()

    if 'Due Date' in df_merged.columns:
        df_merged['Due Date'] = pd.to_datetime(df_merged['Due Date'], errors='coerce')
    else:
        print("Warning: No 'Due Date' column found.")
        return df_merged

    group_cols = ['Item', 'Ship Via Category', 'Ship From', 'destination']

    missing_cols = [col for col in group_cols if col not in df_merged.columns]
    if missing_cols:
        print(f"Warning: Missing grouping columns: {missing_cols}")
        return df_merged

    numeric_cols_to_sum = ['Qty', 'Total Material Cost (Baht)']
    sum_cols = [col for col in numeric_cols_to_sum if col in df_merged.columns]

    indices_to_drop = []

    for group_key, group_indices in df_merged.groupby(group_cols).groups.items():
        if len(group_indices) <= 1:
            continue

        group_df = df_merged.loc[group_indices].copy()
        group_df = group_df.sort_values('Due Date')
        sorted_indices = group_df.index.tolist()

        i = 0
        while i < len(sorted_indices):
            base_idx = sorted_indices[i]
            base_date = group_df.loc[base_idx, 'Due Date']

            window_indices = [base_idx]
            j = i + 1
            while j < len(sorted_indices):
                check_idx = sorted_indices[j]
                check_date = group_df.loc[check_idx, 'Due Date']
                days_diff = (check_date - base_date).days

                if days_diff <= 14:
                    window_indices.append(check_idx)
                    j += 1
                else:
                    break

            if len(window_indices) == 1:
                i += 1
                continue

            # Merge into earliest row
            earliest_idx = base_idx
            other_indices = [idx for idx in window_indices if idx != earliest_idx]

            for col in sum_cols:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
                    sum_value = df_merged.loc[other_indices, col].sum()
                    df_merged.loc[earliest_idx, col] += sum_value

            indices_to_drop.extend(other_indices)
            i = j

    if indices_to_drop:
        indices_to_drop = list(set(indices_to_drop))
        df_merged = df_merged.drop(index=indices_to_drop)
        print(f"✅ Consolidated {len(indices_to_drop)} redundant orders.")

    return df_merged


# ==================== Machine Learning Inference ====================
def predict_import_cost(vendor_name, ship_from, ship_via, item, total_material_cost, unit_price, incoterm, use_model='B'):
    """
    Executes multi-target XGBoost inference for landed cost components.
    """
    ship_from_via = f"{ship_from}_{ship_via}"
    vendor_from_via = f"{vendor_name}_{ship_from_via}"
    vendor_item = f"{vendor_name}_{item}"

    # Incoterm rules
    zero_cols = INCOTERM_RULES.get(incoterm, [])
    exwork_is_zero = 1 if 'Exwork(M)' in zero_cols else 0
    freight_is_zero = 1 if 'Freight(O)' in zero_cols else 0
    local_is_zero = 1 if 'Local(Q)' in zero_cols else 0
    brokerage_is_zero = 1 if 'Brokerage(S)' in zero_cols else 0

    input_m1 = pd.DataFrame({
        'Vendor_From_Via': [vendor_from_via],
        'Incoterm': [incoterm],
        'Item': [item],
        'Total_Material_Cost': [np.log1p(total_material_cost)],
        'Unit_Price': [np.log1p(unit_price)],
        'Exwork_is_zero': [exwork_is_zero],
        'Freight_is_zero': [freight_is_zero],
        'Local_is_zero': [local_is_zero],
        'Brokerage_is_zero': [brokerage_is_zero],
    })

    for col in ['Vendor_From_Via', 'Incoterm', 'Item']:
        input_m1[col] = input_m1[col].astype('category')

    results = {}

    # Predict Freight, Local, Brokerage
    for target in ['Freight(O)', 'Local(Q)', 'Brokerage(S)']:
        if target in models:
            results[target] = np.expm1(models[target].predict(input_m1)).clip(min=0)[0]
        else:
            results[target] = 0.0

    # Predict Exwork
    if exwork_is_zero == 1:
        results['Exwork(M)'] = 0.0
    elif use_model == 'A' and 'Exwork(M)' in models:
        results['Exwork(M)'] = np.expm1(models['Exwork(M)'].predict(input_m1)).clip(min=0)[0]
    elif use_model == 'B' and model2 is not None:
        input_m2 = pd.DataFrame({
            'Vendor_Item': [vendor_item],
            'Total_Material_Cost': [np.log1p(total_material_cost)],
            'Vendor_From_Via': [vendor_from_via],
            'Item': [item],
            'Unit_Price': [np.log1p(unit_price)]
        })
        for col in ['Vendor_Item', 'Vendor_From_Via', 'Item']:
            input_m2[col] = input_m2[col].astype('category')
        results['Exwork(M)'] = np.expm1(model2.predict(input_m2)).clip(min=0)[0]
    else:
        results['Exwork(M)'] = 0.0

    # Enforce zero rules
    for col in zero_cols:
        results[col] = 0.0

    results['Total_Import_cost(U)'] = sum(results.get(t, 0) for t in TARGETS)

    return results


# ==================== Sourcing Evaluation Engine ====================
def find_best_combination(vendor_name, ship_from, item, total_material_cost, unit_price, qty, ship_via_cat, destination, incoterm, use_model='B'):
    """
    Evaluates best shipping method for the given constraints.
    """
    via_map = {
        'SHIP': ['SEA'],
        'AIR': ['AIR', 'FED', 'DHL'],
    }
    via_options = via_map.get(ship_via_cat, ['SEA'])

    best_result = None
    best_total = float('inf')

    for ship_via in via_options:
        try:
            results = predict_import_cost(
                vendor_name=vendor_name,
                ship_from=ship_from,
                ship_via=ship_via,
                item=item,
                total_material_cost=total_material_cost,
                unit_price=unit_price,
                incoterm=incoterm,
                use_model=use_model
            )
            total_cost = results['Total_Import_cost(U)']

            if total_cost < best_total:
                best_total = total_cost
                best_result = {
                    'ship_via': ship_via,
                    'incoterm': incoterm,
                    'exwork': results.get('Exwork(M)', 0.0),
                    'freight': results.get('Freight(O)', 0.0),
                    'local': results.get('Local(Q)', 0.0),
                    'brokerage': results.get('Brokerage(S)', 0.0),
                    'total_cost': total_cost
                }
        except Exception as e:
            continue  # Skip failing combinations

    return best_result


# ==================== Main Batch Pipeline ====================
def process_excel(df, use_model='B'):
    """
    Full pipeline: preprocess → consolidate → predict best costs.
    """
    if df is None or df.empty:
        raise ValueError("Empty input dataframe")

    processed_df = data_preprocessing(df)
    
    print("\n--- Merging duplicate rows ---")
    processed_df = merge_duplicate_rows(processed_df)

    results_list = []
    print("\n--- Finding best shipping combinations ---")

    for idx, row in processed_df.iterrows():
        try:
            ship_via_cat = row.get('Ship Via Category', 'SHIP')
            destination = row.get('destination', 'Thailand')
            incoterm = row.get('Incoterm', 'EXW')

            best = find_best_combination(
                vendor_name=row.get('Vendor Name', ''),
                ship_from=row.get('Ship From', ''),
                item=row.get('Item', ''),
                total_material_cost=float(row.get('Total Material Cost (Baht)', 0)),
                unit_price=float(row.get('Unit Price With Surcharge', 0)),
                qty=float(row.get('Qty', 0)),
                ship_via_cat=ship_via_cat,
                destination=destination,
                incoterm=incoterm,
                use_model=use_model
            )

            if best is None:
                continue

            total_mat_cost = float(row.get('Total Material Cost (Baht)', 0))
            base = total_mat_cost if total_mat_cost > 0 else 1.0

            result_row = row.to_dict()
            result_row['Destination'] = destination
            result_row['Recommended Ship Via'] = best['ship_via']
            result_row['Fixed Syteline Incoterm'] = best['incoterm']

            # ML predictions
            result_row['Predicted Exwork (Baht)'] = round(best['exwork'], 2)
            result_row['Predicted Freight (Baht)'] = round(best['freight'], 2)
            result_row['Predicted Local (Baht)'] = round(best['local'], 2)
            result_row['Predicted Brokerage (Baht)'] = round(best['brokerage'], 2)
            result_row['Predicted Total Import Cost (Baht)'] = round(best['total_cost'], 2)

            # Percentage metrics
            result_row['% Exwork'] = round(best['exwork'] / base, 6)
            result_row['% Freight'] = round(best['freight'] / base, 6)
            result_row['% Local'] = round(best['local'] / base, 6)
            result_row['% Brokerage'] = round(best['brokerage'] / base, 6)
            result_row['% Total Import Cost'] = round(best['total_cost'] / base, 6)

            results_list.append(result_row)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

    result_df = pd.DataFrame(results_list)

    # Clean helper columns
    cols_to_drop = ['Ship Via Category', 'destination']
    for col in cols_to_drop:
        if col in result_df.columns:
            result_df = result_df.drop(columns=[col])

    print(f"✅ Processing complete! Generated {len(result_df)} optimized rows.")
    return result_df
