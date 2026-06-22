# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from cost_predictor import process_excel

# Set wide viewport for scannable multi-column metric alignment
st.set_page_config(
    page_title="VT Garment - Landed Cost Optimization Engine",
    layout="wide"
)

st.title("Optimization Engine")
st.caption("Version 3.0 Dashboard | Business Intelligence Decision Support Portal")
st.markdown("---")

# ==================== SIDEBAR: GLOBAL POLICY VARIABLES ====================
st.sidebar.header("Global Policy Variables")
st.sidebar.markdown("Override standard corporate constraints to test alternative sourcing risk profiles.")

# Financial Rate Parameters
carrying_rate_pct = st.sidebar.slider("Annual Carrying Rate (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
opportunity_rate_pct = st.sidebar.slider("Capital Opportunity Rate (WACC %)", min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100.0
C_RATE = carrying_rate_pct / 100.0
O_RATE = opportunity_rate_pct

# Logistics baseline formula definitions
st.sidebar.markdown("---")
st.sidebar.subheader("Baseline Freight Matrix")
base_fee = st.sidebar.number_input("Fixed Base Fee (THB)", min_value=0, value=16000)
fee_per_shipment = st.sidebar.number_input("Fee Per Shipment (THB)", min_value=0, value=10000)
moq_threshold = st.sidebar.number_input("Supplier Order MOQ Penalty Threshold (THB)", min_value=0.0, value=100000.0)

# Model architecture select
use_model = st.sidebar.selectbox("Inference Model Engine", options=['B', 'A'], index=0)


# ==================== WHAT-IF SIMULATION ENGINE ====================
def run_what_if_analysis(payload, req_date):
    """
    Simulates time-value of money constraints and MOQ penalties 
    over a custom manual shipment schedule array.
    """
    rows = []
    shipping_total = 0.0
    excess_total = 0.0
    carrying_total = 0.0
    opportunity_total = 0.0
    
    for i, ship in enumerate(payload["shipments"]):
        try:
            due_dt = datetime.strptime(str(ship["due_date"]).split(), "%Y-%m-%d")
        except Exception:
            due_dt = req_date
            
        val = float(ship.get("planned_value", 0.0))
        
        # Calculate timeline delays
        days_early = (req_date - due_dt).days
        if days_early < 0:
            return {
                "status": "rejected",
                "message": f"Shipment {i+1} violates timeline rules: Due Date cannot sit after the Material Required Date (Late risk)."
            }
            
        # Evaluate operational minimum order structures
        effective_val = max(val, payload["supplier_moq"])
        excess = effective_val - val
        
        # Sourcing baseline proxy calculation
        shipping_cost = effective_val * 0.045
        
        carrying = effective_val * (C_RATE / 365.0) * days_early
        opportunity = val * (O_RATE / 365.0) * days_early
        
        shipping_total += shipping_cost
        excess_total += excess
        carrying_total += carrying
        opportunity_total += opportunity
        
        rows.append({
            "Shipment": i + 1,
            "Days Early": days_early,
            "Value (THB)": val,
            "MOQ Penalty": excess,
            "Shipping Cost": round(shipping_cost, 2),
            "Carrying Cost": round(carrying, 2),
            "Opportunity Cost": round(opportunity, 2),
            "Total Sourcing Cost": round(shipping_cost + excess + carrying + opportunity, 2)
        })
        
    return {
        "status": "success",
        "grand_total": round(shipping_total + excess_total + carrying_total + opportunity_total, 2),
        "breakdown": {
            "shipping": shipping_total, 
            "moq_excess": excess_total, 
            "carrying": carrying_total, 
            "opportunity": opportunity_total
        },
        "table": rows
    }


# ==================== INTERACTION WORKSPACE ====================
tab1, tab2 = st.tabs(["Automated ERP Manifest Upload", "Interactive Manual Scenario Builder"])

# --- TAB 1: BATCH EXCEL PROCESSING ---
with tab1:
    st.subheader("Upload Syteline Demand Manifest")
    uploaded_file = st.file_uploader("Drop Syteline planning sheets (.xlsx) directly into the optimization pipeline:", type=["xlsx"], key="excel_uploader")

    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file)
        st.markdown("---")
        st.subheader("Current Pipeline Target View")
        st.dataframe(df_raw.head(5), use_container_width=True)
        
        if st.button("Run Sourcing Matrix Optimization", type="primary", key="btn_run_excel"):
            with st.spinner("Executing structural XGBoost inference loops and evaluating import costs..."):
                try:
                    result_df = process_excel(df_raw, use_model=use_model)
                    st.success("Analysis Complete!")
                    
                    # Core aggregations aligned to backend output keys
                    total_import_costs = result_df['Predicted Total Import Cost (Baht)'].sum()
                    total_freight = result_df['Predicted Freight (Baht)'].sum()
                    total_local = result_df['Predicted Local (Baht)'].sum()
                    total_brokerage = result_df['Predicted Brokerage (Baht)'].sum()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Predicted Total Import Cost", f"THB {total_import_costs:,.2f}")
                    c2.metric("Total Predicted Freight", f"THB {total_freight:,.2f}")
                    c3.metric("Total Local Costs", f"THB {total_local:,.2f}")
                    c4.metric("Total Brokerage Costs", f"THB {total_brokerage:,.2f}")
                    
                    st.markdown("---")
                    st.subheader("Individual Shipment Ledger")
                    
                    # Columns restructured to match actual keys assigned inside cost_predictor.py
                    presentation_cols = [
                        'Item', 'Vendor Name', 'Ship From', 'Fixed Syteline Incoterm', 
                        'Recommended Ship Via', 'Predicted Exwork (Baht)', 'Predicted Freight (Baht)', 
                        'Predicted Local (Baht)', 'Predicted Brokerage (Baht)', 
                        'Predicted Total Import Cost (Baht)'
                    ]
                    
                    display_df = result_df[[c for c in presentation_cols if c in result_df.columns]]
                    st.dataframe(display_df, use_container_width=True)
                    
                    csv_data = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Export Optimized Procurement Ledger (.CSV)",
                        data=csv_data,
                        file_name="VTG_Optimized_Sourcing_Plan.csv",
                        mime="text/csv"
                    )
                except Exception as error:
                    st.error(f"Execution Error within pipeline constraints: {str(error)}")
    else:
        st.info("Awaiting Syteline Excel manifest deployment to populate corporate decision metrics.")

# --- TAB 2: INTERACTIVE WHAT-IF SCENARIO BUILDER ---
with tab2:
    st.subheader("Define Your Custom Consolidation Schedule")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        factory_select = st.selectbox("Destination Factory", ["Thailand", "MM (Myanmar)"])
    with col_param2:
        req_date_input = st.date_input("Material Required Date", datetime(2026, 12, 10))
    with col_param3:
        moq_input = st.number_input("Supplier MOQ (THB)", min_value=0.0, value=moq_threshold)
        
    st.markdown("#### Edit Shipment Breakdown Matrix")
    default_schedule = pd.DataFrame([
        {"due_date": "2026-09-10", "planned_value": 200000.0, "vendor_name": "KINGWHALE", "ship_from": "TAIWAN", "ship_via": "SEA", "item": "CKN", "incoterm": "FOB"},
        {"due_date": "2026-11-10", "planned_value": 120000.0, "vendor_name": "KINGWHALE", "ship_from": "TAIWAN", "ship_via": "SEA", "item": "CKN", "incoterm": "FOB"}
    ])
    edited_schedule = st.data_editor(default_schedule, num_rows="dynamic", use_container_width=True, key="schedule_editor")
    
    if st.button("Simulate Sourcing Matrix Scenario", type="primary"):
        req_date_dt = datetime.combine(req_date_input, datetime.min.time())
        simulation_payload = {
            "supplier_moq": moq_input,
            "shipments": edited_schedule.to_dict(orient="records")
        }
        
        results = run_what_if_analysis(simulation_payload, req_date_dt)
        if results["status"] == "rejected":
            st.error(results["message"])
        else:
            st.success("Simulation Metrics Calculated!")
            res1, res2, res3, res4 = st.columns(4)
            res1.metric("Calculated True Sourcing Cost", f"THB {results['grand_total']:,}")
            res2.metric("Simulated Shipping Cost", f"THB {results['breakdown']['shipping']:,}")
            res3.metric("Simulated MOQ Penalties", f"THB {results['breakdown']['moq_excess']:,}")
            res4.metric("Simulated Storage Cost", f"THB {results['breakdown']['carrying']:,}")
            
            st.markdown("---")
            st.subheader("Simulated Ledger Output")
            st.dataframe(pd.DataFrame(results["table"]), use_container_width=True)
            
            st.subheader("Human Decision Override")
            st.radio("Action Verdict Options:", ["Approve Sourcing Plan", "Reject and Consolidate Orders Further"])
