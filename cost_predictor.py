# cost_predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import pickle

# ==================== Global Variables & Initialization ====================
print("Loading cost predictor...")

# Load pre-trained machine learning pipeline checkpoints
models = joblib.load('models.pkl')
model2 = joblib.load('model2.pkl')

# Load commercial baseline rules and configuration schemas
with open('incoterm_rules.json', 'r', encoding='utf-8') as f:
    INCOTERM_RULES = json.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    TARGETS = metadata['targets']

# System boundary options
SHIP_VIA_OPTIONS = ['SEA', 'AIR', 'DHL', 'FED']

print("✅ Cost predictor components mapped successfully!")


# ==================== Data Preprocessing Pipeline ====================
def data_preprocessing(df):
    """
    Transforms raw structured Syteline ERP manifests into uniform features.
    Extracts high-level planning constraints and strips metadata pollution.
    """
    if df is None:
        raise ValueError("Please provide valid input data framework")
    
    df_result = df.copy()
    df_result.columns = df_result.columns.str.strip()
    
    # Isolate general transit mode profile from localized strings
    df_result["Ship Via Category"] = df_result["Ship From"].apply(
        lambda x: "AIR" if "BY AIR" in str(x) else "SHIP"
    )
    
    # Categorize distribution target zones
    df_result["destination"] = df_result["Ship From"].apply(
        lambda x: "MM" if "TO MM" in str(x) else "Thailand"
    )
    
    # Sanitize geographical origin features to stabilize label mappings
    df_result["Ship From"] = df_result["Ship From"].str.replace("BY AIR", "", case=False, regex=False)
    df_result["Ship From"] = df_result["Ship From"].str.replace("TO MM", "", case=False, regex=False)
    df_result["Ship From"] = df_result["Ship From"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df_result


# ==================== Demand Consolidation Engine ====================
def merge_duplicate_rows(df):
    """
    Consolidates split SKU purchase requests tracking on identical logistical lanes 
    within an active 14-day supply planning optimization window.
    """
    if df is None or df.empty:
        return df
    
    df_merged = df.copy()
    
    if 'Due Date' in df_merged.columns:
        df_merged['Due Date'] = pd.to_datetime(df_merged['Due Date'])
    else:
        print("Warning: No 'Due Date' column found, cannot apply 14-day window constraint.")
        return df_merged
    
    # Structural keys required to establish unique procurement lanes
    group_cols = ['Item', 'Ship Via Category', 'Ship From', 'destination']
    
    missing_cols = [col for col in group_cols if col not in df_merged.columns]
    if missing_cols:
        print(f"Warning: Discovered missing keys for lane mapping: {missing_cols}")
        return df
    
    # Identify financial and volumetric quantitative layers to accumulate
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
            
            earliest_idx = base_idx
            other_indices = [idx for idx in window_indices if idx != earliest_idx]
            
            for col in sum_cols:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
                    sum_value = df_merged.loc[other_indices, col].sum()
                    df_merged.loc[earliest_idx, col] = df_merged.loc[earliest_idx, col] + sum_value
            
            indices_to_drop.extend(other_indices)
            print(f"Merged {len(other_indices)} rows into base row (Date: {df_merged.loc[earliest_idx, 'Due Date'].date()}), Group: {group_key}")
            i = j
        
    if indices_to_drop:
        indices_to_drop = list(set(indices_to_drop))
        df_merged = df_merged.drop(index=indices_to_drop)
        print(f"✅ Consolidated {len(indices_to_drop)} redundant orders; {len(df_merged)} primary records retained.")
    else:
        print("Logistical manifest contains no aggregatable windows.")
    
    return df_merged


# ==================== Machine Learning Inference Loop ====================
def predict_import_cost(vendor_name, ship_from, ship_via, item, total_material_cost, unit_price, incoterm, use_model='B'):
    """
    Executes core multi-target machine learning inference using XGBoost.
    Maps underlying item values and lanes to split costs (Exworks, Freight, Local, Brokerage).
    """
    ship_from_via = f"{ship_from}_{ship_via}"
    vendor_from_via = f"{vendor_name}_{ship_from_via}"
    vendor_item = f"{vendor_name}_{item}"
    
    # Map contractual boundaries to explicitly drop commercial legs that are vendor-paid
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
    
    # Predict physical carriage logistics legs via Model 1 Architecture
    for target in ['Freight(O)', 'Local(Q)', 'Brokerage(S)']:
        results[target] = np.expm1(models[target].predict(input_m1)).clip(min=0)
    
    # Evaluate Exworks sourcing overhead using targeted routing architecture
    if exwork_is_zero == 1:
        results['Exwork(M)'] = 0.0
    elif use_model == 'A':
        results['Exwork(M)'] = np.expm1(models['Exwork(M)'].predict(input_m1)).clip(min=0)
    elif use_model == 'B':
        input_m2 = pd.DataFrame({
            'Vendor_Item': [vendor_item],
            'Total_Material_Cost': [np.log1p(total_material_cost)],
            'Vendor_From_Via': [vendor_from_via],
            'Item': [item],
            'Unit_Price': [np.log1p(unit_price)]
        })
        for col in ['Vendor_Item', 'Vendor_From_Via', 'Item']:
            input_m2[col] = input_m2[col].astype('category')
        results['Exwork(M)'] = np.expm1(model2.predict(input_m2)).clip(min=0)
    
    # Hard-enforce zero allocation boundaries mapped from Incoterm schemas
    for col in zero_cols:
        results[col] = 0.0
    
    results['Total_Import_cost(U)'] = sum(results[target] for target in TARGETS)
    
    return results


# ==================== Sourcing Evaluation Engine ====================
def find_best_combination(vendor_name, ship_from, item, total_material_cost, unit_price, qty, ship_via_cat, destination, incoterm, use_model='B'):
    """
    Evaluates shipping speed methods based on routing requirements 
    against a FIXED pre-negotiated Syteline Incoterm.
    """
    SHIP_VIA_OPTIONS = {
        'SHIP': ['SEA'],
        'AIR': ['AIR', 'FED', 'DHL'],
    }
    
    via_options = SHIP_VIA_OPTIONS.get(ship_via_cat, ['SEA'])
    
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
                incoterm=incoterm, # Passed dynamically from current row without modifications
                use_model=use_model
            )
            total_cost = results['Total_Import_cost(U)']
            
            if total_cost < best_total:
                best_total = total_cost
                best_result = {
                    'ship_via': ship_via,
                    'incoterm': incoterm,
                    'exwork': results['Exwork(M)'],
                    'freight': results['Freight(O)'],
                    'local': results['Local(Q)'],
                    'brokerage': results['Brokerage(S)'],
                    'total_cost': results['Total_Import_cost(U)']
                }
        except Exception as e:
            continue
    
    return best_result


# ==================== Batch Execution Pipeline ====================
def process_excel(df, use_model='B'):
    """
    Processes whole input datasets. Maps predictions over existing contract constraints.
    """
    processed_df = data_preprocessing(df)
    
    print("\n--- Merging duplicate rows ---")
    processed_df = merge_duplicate_rows(processed_df)
    
    results_list = []
    print("\n--- Finding best combinations ---")
    for idx, row in processed_df.iterrows():
        try:
            ship_via_cat = row.get('Ship Via Category', 'SHIP')
            destination = row.get('destination', 'Thailand')
            incoterm_from_syteline = row.get('Incoterm', 'EXW')
            
            best = find_best_combination(
                vendor_name=row.get('Vendor Name', ''),
                ship_from=row.get('Ship From', ''),
                item=row.get('Item', ''),
                total_material_cost=float(row.get('Total Material Cost (Baht)', 0)),
                unit_price=float(row.get('Unit Price With Surcharge', 0)),
                qty=float(row.get('Qty', 0)),
                ship_via_cat=ship_via_cat,
                destination=destination,
                incoterm=incoterm_from_syteline, # Locked Parameter
                use_model=use_model
            )
            
            if best is None:
                continue
            
            total_mat_cost = float(row.get('Total Material Cost (Baht)', 0))
            base = total_mat_cost if total_mat_cost > 0 else 1
            
            result_row = row.to_dict()
            result_row['Destination'] = destination
            result_row['Recommended Ship Via'] = best['ship_via']
            result_row['Fixed Syteline Incoterm'] = best['incoterm']
            
            # Appending Pure Machine Learning Output Estimations
            result_row['Predicted Exwork (Baht)'] = round(best['exwork'], 2)
            result_row['Predicted Freight (Baht)'] = round(best['freight'], 2)
            result_row['Predicted Local (Baht)'] = round(best['local'], 2)
            result_row['Predicted Brokerage (Baht)'] = round(best['brokerage'], 2)
            result_row['Predicted Total Import Cost (Baht)'] = round(best['total_cost'], 2)
            
            # Exposure Metric Footprints
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
    
    cols_to_drop = ['Ship Via Category', 'destination']
    for col in cols_to_drop:
        if col in result_df.columns:
            result_df = result_df.drop(columns=[col])
            
    print(f"\n✅ Processing complete! {len(result_df)} rows after merging and prediction.")
    
    return result_df
