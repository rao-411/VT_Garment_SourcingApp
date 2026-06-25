import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# ==================== Load Models ====================
MODEL_LOADED = False
try:
    models = joblib.load('models.pkl')
    model2 = joblib.load('model2.pkl')
    with open('incoterm_rules.json', 'r', encoding='utf-8') as f:
        INCOTERM_RULES = json.load(f)
    MODEL_LOADED = True
    print("✅ ML models loaded successfully.")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    print("   → Running in FALLBACK mode: costs estimated from material value ratios.")
    models, model2, INCOTERM_RULES = {}, None, {}


# ==================== Helper Functions ====================
def data_preprocessing(df):
    if df is None or df.empty:
        return df
    df_res = df.copy()
    df_res.columns = [col.strip() for col in df_res.columns]

    # Safe Ship From handling
    ship_from_col = next((col for col in df_res.columns if 'ship from' in col.lower() or col.lower() == 'origin'), None)
    if ship_from_col is None:
        df_res['Ship From'] = ''
        ship_from_col = 'Ship From'

    ship_from_series = df_res[ship_from_col].fillna('').astype(str)

    df_res['Ship Via Category'] = ship_from_series.apply(
        lambda x: 'AIR' if any(word in x.upper() for word in ['BY AIR', 'DHL', 'FEDEX', 'FED']) else 'SHIP')

    df_res['destination'] = ship_from_series.apply(
        lambda x: 'MM' if 'TO MM' in x.upper() else 'Thailand')

    df_res['Ship From'] = ship_from_series.str.replace('BY AIR', '', case=False).str.strip()

    return df_res


def merge_duplicate_rows(df):
    if df is None or df.empty: return df
    df_m = df.copy()
    df_m['Shipments Combined'] = 1

    date_col = 'PR Delivery Date' if 'PR Delivery Date' in df_m.columns else 'Due Date'
    if date_col in df_m.columns:
        df_m[date_col] = pd.to_datetime(df_m[date_col], errors='coerce')

    # Safe group columns - only use columns that exist
    group_cols = []
    for col in ['Item', 'RMA Description', 'Ship Via Category', 'Ship From', 'destination']:
        if col in df_m.columns:
            group_cols.append(col)

    if not group_cols:
        return df_m  # fallback if no grouping possible

    sum_cols = ['Qty', 'Total Material Cost (Baht)', 'Shipments Combined']
    indices_to_drop = []

    for k, group_indices in df_m.groupby(group_cols).groups.items():
        if len(group_indices) <= 1: continue
        group_df = df_m.loc[group_indices].sort_values(date_col)
        sorted_idx = group_df.index.tolist()
        i = 0
        while i < len(sorted_idx):
            base_idx = sorted_idx[i]
            base_date = group_df.loc[base_idx, date_col]
            window = [base_idx]
            j = i + 1
            while j < len(sorted_idx):
                check_idx = sorted_idx[j]
                if pd.notna(base_date) and pd.notna(group_df.loc[check_idx, date_col]):
                    if (group_df.loc[check_idx, date_col] - base_date).days <= 7:
                        window.append(check_idx)
                        j += 1
                        continue
                break
            if len(window) > 1:
                for col in sum_cols:
                    if col in df_m.columns:
                        df_m.loc[base_idx, col] = pd.to_numeric(df_m.loc[base_idx, col], errors='coerce') + \
                                                  pd.to_numeric(df_m.loc[window[1:], col], errors='coerce').sum()
                indices_to_drop.extend(window[1:])
            i = j

    if indices_to_drop:
        df_m.drop(index=list(set(indices_to_drop)), inplace=True)
    return df_m


def calculate_moq_penalty(group, moq=100000):
    ship_mat_cost = group['Total Material Cost (Baht)'].sum()
    if 'MOQ' in group.columns:
        moq_value = pd.to_numeric(group['MOQ'], errors='coerce').max()
        if pd.notna(moq_value) and moq_value > 0:
            return max(0, moq_value - ship_mat_cost)
    return max(0, moq - ship_mat_cost)


def calculate_financial_costs(shipment_value, carrying_rate, opportunity_rate, days_early=30):
    if shipment_value <= 0 or days_early <= 0:
        return 0.0, 0.0, 0.0
    opp_cost = shipment_value * ((1 + opportunity_rate) ** (days_early / 365) - 1)
    carry_cost = (shipment_value / 2) * carrying_rate * (days_early / 365)
    return round(carry_cost, 2), round(opp_cost, 2), round(carry_cost + opp_cost, 2)


# ==================== ML Functions ====================
def predict_import_cost(vendor_name, ship_from, ship_via, item, total_material_cost, unit_price, incoterm,
                        use_model='B'):
    try:
        ship_from_via = f"{ship_from}_{ship_via}"
        vendor_from_via = f"{vendor_name}_{ship_from_via}"
        vendor_item = f"{vendor_name}_{item}"

        zero_cols = INCOTERM_RULES.get(incoterm, [])

        input_m1 = pd.DataFrame({
            'Vendor_From_Via': [vendor_from_via],
            'Incoterm': [incoterm],
            'Item': [item],
            'Total_Material_Cost': [np.log1p(total_material_cost)],
            'Unit_Price': [np.log1p(unit_price)],
            'Exwork_is_zero': [1 if 'Exwork(M)' in zero_cols else 0],
            'Freight_is_zero': [1 if 'Freight(O)' in zero_cols else 0],
            'Local_is_zero': [1 if 'Local(Q)' in zero_cols else 0],
            'Brokerage_is_zero': [1 if 'Brokerage(S)' in zero_cols else 0]
        })
        for col in ['Vendor_From_Via', 'Incoterm', 'Item']:
            input_m1[col] = input_m1[col].astype('category')

        results = {}
        for t in ['Freight(O)', 'Local(Q)', 'Brokerage(S)']:
            if t in models and models[t] is not None:
                results[t] = np.expm1(models[t].predict(input_m1)).clip(min=0)[0]
            else:
                results[t] = 0.0

        if 'Exwork(M)' in zero_cols:
            results['Exwork(M)'] = 0.0
        elif model2 is not None:
            input_m2 = pd.DataFrame({
                'Vendor_Item': [vendor_item],
                'Total_Material_Cost': [np.log1p(total_material_cost)],
                'Vendor_From_Via': [vendor_from_via],
                'Item': [item],
                'Unit_Price': [np.log1p(unit_price)]
            })
            for c in ['Vendor_Item', 'Vendor_From_Via', 'Item']:
                input_m2[c] = input_m2[c].astype('category')
            results['Exwork(M)'] = np.expm1(model2.predict(input_m2)).clip(min=0)[0]
        else:
            results['Exwork(M)'] = 0.0

        return {
            'Exwork(M)': results.get('Exwork(M)', 0),
            'Freight(O)': results.get('Freight(O)', 0),
            'Local(Q)': results.get('Local(Q)', 0),
            'Brokerage(S)': results.get('Brokerage(S)', 0),
            'Total_Import_cost(U)': sum(results.values())
        }
    except:
        return {'Exwork(M)': 0, 'Freight(O)': 0, 'Local(Q)': 0, 'Brokerage(S)': 0, 'Total_Import_cost(U)': 0}


def find_best_combination(vendor_name, ship_from, item, total_material_cost, unit_price, qty, ship_via_cat, destination,
                          incoterm, use_model='B'):
    via_opts = ['SEA'] if ship_via_cat == 'SHIP' else ['AIR', 'FED', 'DHL']
    best_res = None
    best_tot = float('inf')
    for via in via_opts:
        try:
            r = predict_import_cost(vendor_name, ship_from, via, item, total_material_cost, unit_price, incoterm,
                                    use_model)
            if r['Total_Import_cost(U)'] < best_tot:
                best_tot = r['Total_Import_cost(U)']
                best_res = {
                    'ship_via': via,
                    'incoterm': incoterm,
                    'exwork': r['Exwork(M)'],
                    'freight': r['Freight(O)'],
                    'local': r['Local(Q)'],
                    'brokerage': r['Brokerage(S)']
                }
        except:
            continue
    return best_res


# ==================== Main Process ====================
def process_excel(df, use_model='B', c_rate=0.06, o_rate=0.10, base_fee=16000, ship_fee=10000):
    if df is None or df.empty:
        return pd.DataFrame()

    pdf = data_preprocessing(df)
    pdf = merge_duplicate_rows(pdf)
    res_list = []

    shipment_groups = pdf.groupby(['Vendor Name', 'Ship From', 'Due Date', 'Ship Via Category'])

    for name, group in shipment_groups:
        v_name, s_from, d_date, s_via_cat = name
        ship_mat_cost = group['Total Material Cost (Baht)'].sum()

        dom_row = group.loc[group['Total Material Cost (Baht)'].idxmax()]
        incoterm = dom_row.get('Incoterm', 'EXW')
        u_price = dom_row.get('Unit Price With Surcharge', 0)

        try:
            best = find_best_combination(v_name, s_from, dom_row.get('Item', 'Unknown'), ship_mat_cost,
                                         u_price, group['Qty'].sum(), s_via_cat, None, incoterm, use_model)

            # FIX: if models not loaded, ML silently returns all zeros → detect and use fallback ratios
            if not MODEL_LOADED or best is None or (best.get('exwork', 0) + best.get('local', 0) + best.get('brokerage', 0) == 0):
                s_ex = ship_mat_cost * 0.025   # ~2.5% ex-works surcharge
                s_fr = ship_mat_cost * 0.05  # ~5% freight (pure carrier cost, excl. fixed logistics fees)
                s_lo = ship_mat_cost * 0.012   # ~1.2% local trucking & handling
                s_br = ship_mat_cost * 0.008 + 1500  # ~0.8% + base brokerage fee
                f_via = best['ship_via'] if best else s_via_cat
                f_inc = f"{incoterm} [ML-Fallback]"
            else:
                s_fr = best['freight']  # pure ML-predicted carrier freight, excl. fixed logistics fees
                s_ex = best.get('exwork', 0)
                s_lo = best.get('local', 0)
                s_br = best.get('brokerage', 0)
                f_via = best['ship_via']
                f_inc = best['incoterm']
        except:
            s_ex = ship_mat_cost * 0.025
            s_fr = ship_mat_cost * 0.05  # ~5% freight (pure carrier cost)
            s_lo = ship_mat_cost * 0.012
            s_br = ship_mat_cost * 0.008 + 1500
            f_via, f_inc = s_via_cat, f"{incoterm} [ML-Fallback]"

        s_moq_pen = calculate_moq_penalty(group)

        for idx, row in group.iterrows():
            r_dict = row.to_dict()
            l_cost = float(row.get('Total Material Cost (Baht)', 0))
            ratio = l_cost / ship_mat_cost if ship_mat_cost > 0 else 0

            req_date = row.get('Due Date')
            ship_date = row.get('PR Delivery Date', req_date)
            try:
                days_early = max(0, (pd.to_datetime(req_date) - pd.to_datetime(ship_date)).days)
            except:
                days_early = 30


            r_dict['Consolidation Status'] = 'Grouped in Shipment' if len(group) > 1 else 'Single'
            r_dict['Recommended Ship Via'] = f_via
            r_dict['Fixed Syteline Incoterm'] = f_inc
            r_dict['Predicted Exwork (Baht)'] = round(s_ex * ratio, 2)
            r_dict['Predicted Freight (Baht)'] = round(s_fr * ratio, 2)
            r_dict['Predicted Local (Baht)'] = round(s_lo * ratio, 2)
            r_dict['Predicted Brokerage (Baht)'] = round(s_br * ratio, 2)
            r_dict['MOQ Penalty (Baht)'] = round(s_moq_pen * ratio, 2)

            r_dict['Predicted Total Import Cost (Baht)'] = round(
                r_dict.get('Predicted Exwork (Baht)', 0) +
                r_dict.get('Predicted Freight (Baht)', 0) +
                r_dict.get('Predicted Local (Baht)', 0) +
                r_dict.get('Predicted Brokerage (Baht)', 0) +
                r_dict.get('MOQ Penalty (Baht)', 0) , 2)

            res_list.append(r_dict)

    rdf = pd.DataFrame(res_list)
    for c in ['Ship Via Category', 'destination']:
        if c in rdf.columns:
            rdf.drop(columns=[c], inplace=True, errors='ignore')
    return rdf


# ==================== Tab 2: Scenario Simulator ====================
def evaluate_sourcing_scenarios(total_material_value, shipment_due_date, material_required_date,
                                moq, fixed_base_fee, fee_per_shipment, c_rate, o_rate, interval_days):
    results = []
    req_date = pd.to_datetime(material_required_date)

    for num_shipments in [1, 2, 4]:
        base_value = total_material_value / num_shipments
        total_shipping = fixed_base_fee + (num_shipments * fee_per_shipment)
        total_carry = 0
        total_opp = 0
        total_moq_penalty = 0

        for i in range(num_shipments):
            due_date = pd.to_datetime(shipment_due_date) + pd.Timedelta(days=i * interval_days)
            days_early = max(0, (req_date - due_date).days)

            effective_value = max(base_value, moq)
            moq_penalty = effective_value - base_value
            total_moq_penalty += moq_penalty

            carry, opp, _ = calculate_financial_costs(effective_value, c_rate, o_rate, days_early)
            total_carry += carry
            total_opp += opp

        total_cost = total_material_value + total_moq_penalty + total_shipping + total_carry + total_opp

        results.append({
            "Scenario": f"{num_shipments} Shipments",
            "Total Freight (THB)": round(total_shipping, 2),
            "MOQ Penalty (THB)": round(total_moq_penalty, 2),
            "Carrying Cost (THB)": round(total_carry, 2),
            "Opportunity Cost (THB)": round(total_opp, 2),
            "True Landed Cost (THB)": round(total_cost, 2)
        })

    return pd.DataFrame(results)
