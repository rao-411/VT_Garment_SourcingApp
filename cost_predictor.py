# cost_predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import pickle
from sklearn.preprocessing import LabelEncoder

# ==================== 全局变量（加载时初始化） ====================
print("Loading cost predictor...")

# 加载模型
models = joblib.load('models.pkl')
model2 = joblib.load('model2.pkl')

# 加载配置
with open('incoterm_rules.json', 'r', encoding='utf-8') as f:
    INCOTERM_RULES = json.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    TARGETS = metadata['targets']

# 可选的 Ship Via 和 Incoterm
SHIP_VIA_OPTIONS = ['SEA', 'AIR', 'DHL', 'FED']
INCOTERM_OPTIONS = ['EXW', 'FOB', 'CIF', 'DAP', 'DDP']

print("✅ Cost predictor loaded successfully!")


# ==================== 数据预处理 ====================
def data_preprocessing(df):
    """将原始Excel数据转换为模型输入格式"""
    if df is None:
        raise ValueError("Please provide input data")
    
    df_result = df.copy()
    df_result.columns = df_result.columns.str.strip()
    
    
    # 提取 Ship Via Category（用于后续优化选项）
    df_result["Ship Via Category"] = df_result["Ship From"].apply(
        lambda x: "AIR" if "BY AIR" in str(x) else "SHIP"
    )
    
    # 提取 destination
    df_result["destination"] = df_result["Ship From"].apply(
        lambda x: "MM" if "TO MM" in str(x) else "Thailand"
    )
    
    # 清理 Ship From
    df_result["Ship From"] = df_result["Ship From"].str.replace("BY AIR", "", case=False, regex=False)
    df_result["Ship From"] = df_result["Ship From"].str.replace("TO MM", "", case=False, regex=False)
    df_result["Ship From"] = df_result["Ship From"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df_result


# ==================== 合并重复数据 ====================
def merge_duplicate_rows(df):
    """
    将相同条件的数据条合并到最早日期的记录中
    
    合并条件：
    - Item 相同
    - Ship Via Category 相同
    - Ship From 相同
    - destination 相同
    
    合并逻辑（按你的思路）：
    1. 按 Due Date 排序
    2. 从最早的日期作为 base
    3. 将 base 之后 14 天内的数据合并到 base
    4. 合并后，base 保持不变
    5. 找到下一个与 base 相差 >14 天的数据作为新的 base
    6. 重复直到该分组结束
    
    示例：
    日期: 1, 13, 25
    - base=1, 13在14天内 → 合并到1
    - 25与1差24天 >14天 → 新的base=25
    - 25后面没有数据 → 结束
    
    结果：保留两行（1和25）
    """
    if df is None or df.empty:
        return df
    
    df_merged = df.copy()
    
    # 确保 Due Date 是 datetime 类型
    if 'Due Date' in df_merged.columns:
        df_merged['Due Date'] = pd.to_datetime(df_merged['Due Date'])
    else:
        print("Warning: No 'Due Date' column found, cannot apply 14-day window constraint.")
        return df_merged
    
    # 定义用于分组的列
    group_cols = ['Item', 'Ship Via Category', 'Ship From', 'destination']
    
    # 检查必要的列是否存在
    missing_cols = [col for col in group_cols if col not in df_merged.columns]
    if missing_cols:
        print(f"Warning: Missing columns for grouping: {missing_cols}")
        return df
    
    # 需要累加的数值列
    numeric_cols_to_sum = ['Qty', 'Total Material Cost (Baht)']
    
    # 实际存在的数值列
    sum_cols = [col for col in numeric_cols_to_sum if col in df_merged.columns]
    
    # 存储要删除的索引
    indices_to_drop = []
    
    # 按分组处理
    for group_key, group_indices in df_merged.groupby(group_cols).groups.items():
        if len(group_indices) <= 1:
            continue
        
        # 获取该分组的所有行，按 Due Date 排序
        group_df = df_merged.loc[group_indices].copy()
        group_df = group_df.sort_values('Due Date')
        sorted_indices = group_df.index.tolist()
        
        # 按你的思路：贪心 + 滑动 base
        i = 0
        while i < len(sorted_indices):
            base_idx = sorted_indices[i]
            base_date = group_df.loc[base_idx, 'Due Date']
            
            # 找到所有在 base 日期 14 天内的行
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
            
            # 如果只有 base 自己，没有其他行在14天内，直接跳到下一个
            if len(window_indices) == 1:
                i += 1
                continue
            
            # base 是最早的，不需要再找 earliest_idx
            earliest_idx = base_idx
            
            # 其他行（除了 base 之外的行）
            other_indices = [idx for idx in window_indices if idx != earliest_idx]
            
            # 将其他行的数值累加到最早的行
            for col in sum_cols:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
                    sum_value = df_merged.loc[other_indices, col].sum()
                    df_merged.loc[earliest_idx, col] = df_merged.loc[earliest_idx, col] + sum_value
            
            # 标记要删除的行
            indices_to_drop.extend(other_indices)
            
            print(f"合并 {len(other_indices)} 行到 base 行 (日期: {df_merged.loc[earliest_idx, 'Due Date'].date()})，分组: {group_key}")
            
            # 移动到下一个 base（窗口结束后的第一个索引）
            i = j
        
    # 删除重复的行
    if indices_to_drop:
        indices_to_drop = list(set(indices_to_drop))
        df_merged = df_merged.drop(index=indices_to_drop)
        print(f"✅ 共合并了 {len(indices_to_drop)} 行，剩余 {len(df_merged)} 行")
    else:
        print("没有需要合并的行")
    
    return df_merged


# ==================== 单行预测（按照 notebook 中 predict_import_cost 的逻辑） ====================
def predict_import_cost(vendor_name, ship_from, ship_via, item, total_material_cost, unit_price, incoterm, use_model='B'):
    """
    完全按照 notebook 中 predict_import_cost 的逻辑
    """
    ship_from_via = f"{ship_from}_{ship_via}"
    vendor_from_via = f"{vendor_name}_{ship_from_via}"
    vendor_item = f"{vendor_name}_{item}"
    
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
    
    # 预测 Freight, Local, Brokerage（用 model1）
    for target in ['Freight(O)', 'Local(Q)', 'Brokerage(S)']:
        results[target] = np.expm1(models[target].predict(input_m1))[0].clip(min=0)
    
    # 预测 Exwork
    if exwork_is_zero == 1:
        results['Exwork(M)'] = 0.0
    elif use_model == 'A':
        results['Exwork(M)'] = np.expm1(models['Exwork(M)'].predict(input_m1))[0].clip(min=0)
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
        results['Exwork(M)'] = np.expm1(model2.predict(input_m2))[0].clip(min=0)
    
    # 根据 incoterm 规则强制某些费用为 0
    for col in zero_cols:
        results[col] = 0.0
    
    results['Total_Import_cost(U)'] = sum(results[target] for target in TARGETS)
    
    return results


# ==================== 找最佳组合（按照 notebook 中 find_best_combinations 的逻辑） ====================
def find_best_combination(vendor_name, ship_from, item, total_material_cost, unit_price, qty, ship_via_cat, destination, use_model='B'):
    """
    按照 notebook 中 find_best_combinations 的逻辑
    注意：notebook 中使用的是 model_MM/model_Thai 两个不同的模型集
    这里简化使用全局 models/model2，实际使用时需要根据 destination 选择模型
    """
    SHIP_VIA_OPTIONS = {
        'SHIP': ['SEA'],
        'AIR': ['AIR', 'FED', 'DHL'],
    }
    INCOTERM_OPTIONS = ['EXW', 'CIF', 'FOB']
    
    via_options = SHIP_VIA_OPTIONS.get(ship_via_cat, ['SEA'])
    incoterm_options = ['EXW'] if ship_from.upper().startswith('CHINA') else INCOTERM_OPTIONS
    
    best_result = None
    best_total = float('inf')
    
    for ship_via in via_options:
        for incoterm in incoterm_options:
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
                        'exwork': results['Exwork(M)'],
                        'freight': results['Freight(O)'],
                        'local': results['Local(Q)'],
                        'brokerage': results['Brokerage(S)'],
                        'total_cost': results['Total_Import_cost(U)']
                    }
            except Exception as e:
                continue
    
    return best_result


# ==================== 批量处理 Excel ====================
def process_excel(df, use_model='B'):
    """
    处理整个 Excel 文件
    输入：原始 DataFrame（用户上传的）
    输出：带预测结果的 DataFrame
    """
    # 步骤1：数据预处理
    processed_df = data_preprocessing(df)
    
    # 步骤2：合并重复数据
    print("\n--- Merging duplicate rows ---")
    processed_df = merge_duplicate_rows(processed_df)
    
    # 步骤3：对每一行进行预测
    results_list = []
    
    print("\n--- Finding best combinations ---")
    for idx, row in processed_df.iterrows():
        try:
            ship_via_cat = row.get('Ship Via Category', 'SHIP')
            destination = row.get('destination', 'Thailand')
            
            best = find_best_combination(
                vendor_name=row.get('Vendor Name', ''),
                ship_from=row.get('Ship From', ''),
                item=row.get('Item', ''),
                total_material_cost=float(row.get('Total Material Cost (Baht)', 0)),
                unit_price=float(row.get('Unit Price With Surcharge', 0)),
                qty=float(row.get('Qty', 0)),
                ship_via_cat=ship_via_cat,
                destination=destination,
                use_model=use_model
            )
            
            if best is None:
                continue
            
            total_mat_cost = float(row.get('Total Material Cost (Baht)', 0))
            base = total_mat_cost if total_mat_cost > 0 else 1
            
            result_row = row.to_dict()
            result_row['Destination'] = destination
            result_row['Recommended Ship Via'] = best['ship_via']
            result_row['Recommended Incoterm'] = best['incoterm']
            result_row['Predicted Exwork (Baht)'] = round(best['exwork'], 2)
            result_row['Predicted Freight (Baht)'] = round(best['freight'], 2)
            result_row['Predicted Local (Baht)'] = round(best['local'], 2)
            result_row['Predicted Brokerage (Baht)'] = round(best['brokerage'], 2)
            result_row['Predicted Total Import Cost (Baht)'] = round(best['total_cost'], 2)
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