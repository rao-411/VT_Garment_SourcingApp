import streamlit as st
import pandas as pd
import plotly.graph_objects as go  # ADD THIS IMPORT AT THE TOP
from cost_predictor import process_excel
from datetime import date

# 页面配置
st.set_page_config(page_title="VT Garment - Optimization Engine", layout="wide")
st.title("Optimization Engine")
st.caption("Version 3.0 Dashboard | Business Intelligence Decision Support Portal")
st.markdown("---")

# 初始化 session state，用于保存运算结果，防止刷新丢失
if 'consolidated_data' not in st.session_state:
    st.session_state['consolidated_data'] = None

# 侧边栏参数 - 全局财务政策
st.sidebar.header("Global Policy Variables")
carrying_rate = st.sidebar.slider("Annual Carrying Rate (%)", 0.0, 20.0, 6.0, step=0.5) / 100
opportunity_rate = st.sidebar.slider("Capital Opportunity Rate (WACC %)", 0.0, 20.0, 10.0, step=0.5) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Baseline Freight Matrix")
base_fee = st.sidebar.number_input("Fixed Base Fee (THB)", value=16000)
fee_per_shipment = st.sidebar.number_input("Fee Per Shipment (THB)", value=10000)

# 核心界面
st.subheader("Automated ERP Manifest Upload & Scenario Optimization")

uploaded_file = st.file_uploader("Drop Syteline planning sheets (.xlsx):", type=["xlsx", "csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    st.markdown("---")
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Run Sourcing Matrix Optimization", type="primary"):
        with st.spinner("Running XGBoost inference & Scenario Simulation..."):

            # 第一阶段：跑 ML 预测并合并
            result_df = process_excel(
                df,
                c_rate=carrying_rate,
                o_rate=opportunity_rate,
                base_fee=base_fee,
                ship_fee=fee_per_shipment
            )

            # 分组聚合逻辑：无视 Item，按相同物流属性和时间整合为“一个集装箱/一个批次”
            groupby_cols = ['Due Date', 'PR Delivery Date', 'Vendor Name', 'Ship From', 'Recommended Ship Via',
                            'Fixed Syteline Incoterm']
            actual_groupby = [c for c in groupby_cols if c in result_df.columns]

            agg_dict = {
                'MOQ': 'max',
                'Qty': 'sum',
                'Total Material Cost (Baht)': 'sum',
                'Consolidation Status': 'first',
                'Shipments Combined': 'sum',
                'Predicted Exwork (Baht)': 'sum',
                'Predicted Freight (Baht)': 'sum',
                'Predicted Local (Baht)': 'sum',
                'Predicted Brokerage (Baht)': 'sum',
                'MOQ Penalty (Baht)': 'sum',
                'Predicted Total Import Cost (Baht)': 'sum'
            }
            actual_agg = {k: v for k, v in agg_dict.items() if k in result_df.columns}

            consolidated_df = result_df.groupby(actual_groupby, as_index=False).agg(actual_agg)


            # 第二阶段：内嵌 Jason 的多场景自动推演逻辑
            def calculate_scenarios(row):
                req_date = pd.to_datetime(row.get('Due Date'), errors='coerce')
                ship_date = pd.to_datetime(row.get('PR Delivery Date'), errors='coerce')

                if pd.isna(ship_date):
                    ship_date = req_date

                moq = row.get('MOQ', 0)
                qty = row.get('Qty', 0)
                moq_qty_gap = max(0, moq - qty)
                total_material_value = row.get('Total Material Cost (Baht)', 0)
                interval_days = 30

                res = {}
                best_cost = float('inf')
                best_scenario = 1

                for s_idx, num_shipments in zip([1, 2, 3], [1, 2, 4]):
                    base_value = total_material_value / num_shipments if num_shipments > 0 else 0

                    total_actual_value = 0
                    total_excess = 0
                    total_opportunity_cost = 0
                    total_carrying_cost = 0

                    for i in range(num_shipments):
                        current_due = ship_date + pd.Timedelta(days=i * interval_days)
                        days_early = (req_date - current_due).days
                        if days_early < 0:
                            days_early = 0

                        effective_value = max(base_value, moq)
                        excess = effective_value - base_value

                        total_actual_value += effective_value
                        total_excess += excess

                        if days_early > 0:
                            total_opportunity_cost += effective_value * opportunity_rate / 365.0 * days_early
                            total_carrying_cost += effective_value * carrying_rate / 365.0 * days_early

                    shipping = base_fee + (fee_per_shipment * num_shipments)
                    total_cost = total_actual_value + shipping + total_opportunity_cost + total_carrying_cost

                    res[f'Scenario {s_idx} cost'] = round(total_cost, 2)
                    res[f'Scenario {s_idx} material'] = round(total_actual_value, 2)
                    res[f'Scenario {s_idx} excess'] = round(total_excess, 2)
                    res[f'Scenario {s_idx} shipping'] = round(shipping, 2)
                    res[f'Scenario {s_idx} carrying'] = round(total_carrying_cost, 2)
                    res[f'Scenario {s_idx} opportunity'] = round(total_opportunity_cost, 2)

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_scenario = s_idx

                res['MOQ-Qty Gap'] = moq_qty_gap
                res['Best scenario number'] = best_scenario
                res['Best cost'] = round(best_cost, 2)

                return pd.Series(res)


            scenario_metrics = consolidated_df.apply(calculate_scenarios, axis=1)
            final_consolidated_df = pd.concat([consolidated_df, scenario_metrics], axis=1)

            # Add "Total Freight (incl. Logistics Fee)" columns per scenario
            # = Predicted Freight (pure carrier cost) + Scenario N shipping (base fee + per-shipment fee)
            for s_idx in [1, 2, 3]:
                if 'Predicted Freight (Baht)' in final_consolidated_df.columns:
                    final_consolidated_df[f'Scenario {s_idx} total freight'] = (
                        final_consolidated_df['Predicted Freight (Baht)'].fillna(0) +
                        final_consolidated_df[f'Scenario {s_idx} shipping'].fillna(0)
                    ).round(2)

            # 【修复点】：将计算好的数据存入 session_state 记忆中
            st.session_state['consolidated_data'] = final_consolidated_df
            st.success("Analysis Complete!")

    # ==============================================================
    # 以下 UI 渲染代码被移出 if st.button() 外面
    # 只要 session_state 里有数据，重新运行时就会继续渲染表格和图表
    # ==============================================================
    if st.session_state['consolidated_data'] is not None:
        final_consolidated_df = st.session_state['consolidated_data']

        st.markdown("---")
        st.subheader("Optimized Consolidated Shipment Ledger")

        # 开启表格的交互式“行选择”功能
        st.markdown(
            "*(Tip: Click the checkbox on the left of any row below to view its detailed multi-scenario cost breakdown)*")
        selection_event = st.dataframe(
            final_consolidated_df,
            use_container_width=True,
            on_select="rerun",  # 选中后触发重新运行
            selection_mode="single-row"  # 单选模式
        )

        csv_data_cons = final_consolidated_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Consolidated Scenario Ledger (.CSV)",
            data=csv_data_cons,
            file_name="VTG_Optimized_Consolidated_Scenarios.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("Consolidated Shipment Multi-Scenario Analysis")

        # 读取选中的行，并绘制图表
        selected_rows = selection_event.selection.rows
        if len(selected_rows) > 0:
            row_idx = selected_rows[0]
            selected_data = final_consolidated_df.iloc[row_idx]

            # 提取这一行的各项成本数据构建图表 DataFrame
            chart_data = pd.DataFrame({
                "Scenario": ["Scenario 1", "Scenario 2", "Scenario 3"],
                "Material": [selected_data["Scenario 1 material"], selected_data["Scenario 2 material"],
                             selected_data["Scenario 3 material"]],
                "MOQ Excess": [selected_data["Scenario 1 excess"], selected_data["Scenario 2 excess"],
                               selected_data["Scenario 3 excess"]],
                "Shipping": [selected_data["Scenario 1 shipping"], selected_data["Scenario 2 shipping"],
                             selected_data["Scenario 3 shipping"]],
                "Carrying Cost": [selected_data["Scenario 1 carrying"], selected_data["Scenario 2 carrying"],
                                  selected_data["Scenario 3 carrying"]],
                "Opportunity Cost": [selected_data["Scenario 1 opportunity"], selected_data["Scenario 2 opportunity"],
                                     selected_data["Scenario 3 opportunity"]]
            }).set_index("Scenario")

            # 在图表上方显示选中行的简要信息
            st.markdown(
                f"**Analyzing Selected Order:** `{selected_data.get('Vendor Name', 'N/A')}` | **Ship From:** `{selected_data.get('Ship From', 'N/A')}` | **Required Date:** `{selected_data.get('Due Date', 'N/A')}`")

            # ==============================================================
            # ENHANCED PLOTLY CHART WITH COLOR OPTIMIZATION
            # ==============================================================

            # Prepare data for plotting
            scenarios = chart_data.index.tolist()
            material = chart_data["Material"].values
            moq_excess = chart_data["MOQ Excess"].values
            shipping = chart_data["Shipping"].values
            carrying = chart_data["Carrying Cost"].values
            opportunity = chart_data["Opportunity Cost"].values

            # Create figure
            fig = go.Figure()

            # Add traces in SPECIFIC ORDER (bottom to top)
            # 1. SHIPPING - Light gray (recedes into background)
            fig.add_trace(go.Bar(
                name='Shipping',
                x=scenarios,
                y=shipping,
                marker_color='#00CC96',  # Light gray
                marker_line_color='white',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>Shipping: ฿%{y:,.2f}<extra></extra>'
            ))

            # 2. MATERIAL - Vivid orange (pulls forward)
            fig.add_trace(go.Bar(
                name='Material',
                x=scenarios,
                y=material,
                marker_color='#FF6B35',  # Bright orange
                marker_line_color='white',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>Material: ฿%{y:,.2f}<extra></extra>'
            ))

            # 3. MOQ EXCESS - Deep pink/red (alerts viewer)
            fig.add_trace(go.Bar(
                name='MOQ Excess',
                x=scenarios,
                y=moq_excess,
                marker_color='#D1345B',  # Deep pink/red
                marker_line_color='white',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>MOQ Excess: ฿%{y:,.2f}<extra></extra>'
            ))

            # 4. CARRYING COST - Deep purple (grounds the top)
            fig.add_trace(go.Bar(
                name='Carrying Cost',
                x=scenarios,
                y=carrying,
                marker_color='#5A189A',  # Deep purple
                marker_line_color='white',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>Carrying Cost: ฿%{y:,.2f}<extra></extra>'
            ))

            # 5. OPPORTUNITY COST - Teal/green (refreshing contrast)
            fig.add_trace(go.Bar(
                name='Opportunity Cost',
                x=scenarios,
                y=opportunity,
                marker_color='#00A896',  # Teal green
                marker_line_color='white',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>Opportunity Cost: ฿%{y:,.2f}<extra></extra>'
            ))

            # Customize layout
            fig.update_layout(
                barmode='stack',
                title={
                    'text': 'Consolidated Shipment Multi-Scenario Analysis',
                    'font': {'size': 20, 'family': 'Arial Black'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis={
                    'title': 'Scenario',
                    'title_font': {'size': 14, 'family': 'Arial'},
                    'tickfont': {'size': 12},
                    'gridcolor': '#000000'
                },
                yaxis={
                    'title': 'Cost (THB)',
                    'title_font': {'size': 14, 'family': 'Arial'},
                    'tickfont': {'size': 12},
                    'gridcolor': '#000000',
                    'tickformat': ',.0f'
                },
                legend={

                    'font': {'size': 11},
                    'orientation': 'h',  # Horizontal legend
                    'yanchor': 'bottom',
                    'y': 1.02,
                    'xanchor': 'center',
                    'x': 0.5,
                    'bgcolor': 'rgba(255,255,255,0.8)',
                    'bordercolor': '#000000',
                    'borderwidth': 1
                },
                hovermode='x unified',
                plot_bgcolor='black',
                paper_bgcolor='black',
                height=500,
                margin=dict(t=80, b=50, l=60, r=40)
            )

            # Optional: Add a subtle horizontal line at the best cost
            best_cost = selected_data['Best cost']
            fig.add_hline(
                y=best_cost,
                line_dash="dash",
                line_color="#FF6B35",
                line_width=2,
                annotation_text=f"Best: ฿{best_cost:,.2f}",
                annotation_font=dict(size=11, color="#FF6B35"),
                annotation_position="top right"
            )

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

        else:
            # 如果没有选中任何行，显示提示语
            st.info("Select a row in the table above to view its scenario breakdown chart.")
