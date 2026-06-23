# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from cost_predictor import process_excel

# Set wide viewport
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
opportunity_rate_pct = st.sidebar.slider("Capital Opportunity Rate (WACC %)", min_value=0.0, max_value=20.0, value=10.0, step=0.5)

C_RATE = carrying_rate_pct / 100.0
O_RATE = opportunity_rate_pct / 100.0

# Logistics baseline
st.sidebar.markdown("---")
st.sidebar.subheader("Baseline Freight Matrix")
base_fee = st.sidebar.number_input("Fixed Base Fee (THB)", min_value=0, value=16000)
fee_per_shipment = st.sidebar.number_input("Fee Per Shipment (THB)", min_value=0, value=10000)
moq_threshold = st.sidebar.number_input("Supplier Order MOQ Penalty Threshold (THB)", min_value=0.0, value=100000.0)

# Model select
use_model = st.sidebar.selectbox("Inference Model Engine", options=['B', 'A'], index=0)

# ==================== WHAT-IF SIMULATION ENGINE ====================
def run_what_if_analysis(payload, req_date):
    """Simulates time-value of money + MOQ penalties."""
    rows = []
    shipping_total = 0.0
    excess_total = 0.0
    carrying_total = 0.0
    opportunity_total = 0.0
    
    for i, ship in enumerate(payload["shipments"]):
        try:
            # Robust date parsing
            due_str = str(ship.get("due_date", "")).strip()
            if isinstance(due_str, date):
                due_dt = datetime.combine(due_str, datetime.min.time())
            else:
                due_dt = datetime.strptime(due_str.split()[0], "%Y-%m-%d")
        except Exception:
            due_dt = req_date
            
        val = float(ship.get("planned_value", 0.0))
        
        # Timeline check
        days_early = (req_date - due_dt).days
        if days_early < 0:
            return {
                "status": "rejected",
                "message": f"Shipment {i+1} violates timeline: Due Date cannot be after Material Required Date."
            }
            
        # MOQ
        effective_val = max(val, payload["supplier_moq"])
        excess = effective_val - val
        
        # Proxy costs (you can enhance this)
        shipping_cost = effective_val * 0.045  # placeholder
        
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

# ==================== MAIN UI ====================
tab1, tab2 = st.tabs(["Automated ERP Manifest Upload", "Interactive Manual Scenario Builder"])

# --- TAB 1: Excel Processing ---
with tab1:
    st.subheader("Upload Syteline Demand Manifest")
    uploaded_file = st.file_uploader("Drop Syteline planning sheets (.xlsx):", type=["xlsx"], key="excel_uploader")

    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file)
        st.markdown("---")
        st.subheader("Preview")
        st.dataframe(df_raw.head(5), use_container_width=True)
        
        if st.button("Run Sourcing Matrix Optimization", type="primary"):
            with st.spinner("Running XGBoost inference + cost optimization..."):
                try:
                    result_df = process_excel(df_raw, use_model=use_model)
                    st.success("Analysis Complete!")
                    
                    total_import = result_df.get('Predicted Total Import Cost (Baht)', pd.Series(0)).sum()
                    total_freight = result_df.get('Predicted Freight (Baht)', pd.Series(0)).sum()
                    total_local = result_df.get('Predicted Local (Baht)', pd.Series(0)).sum()
                    total_brokerage = result_df.get('Predicted Brokerage (Baht)', pd.Series(0)).sum()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Predicted Total Import Cost", f"THB {total_import:,.2f}")
                    c2.metric("Total Predicted Freight", f"THB {total_freight:,.2f}")
                    c3.metric("Total Local Costs", f"THB {total_local:,.2f}")
                    c4.metric("Total Brokerage Costs", f"THB {total_brokerage:,.2f}")
                    
                    st.markdown("---")
                    st.subheader("Optimized Ledger")
                    
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
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
    else:
        st.info("Upload an Excel manifest to start optimization.")

# --- TAB 2: What-If Builder ---
with tab2:
    st.subheader("Custom Consolidation Schedule")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        factory_select = st.selectbox("Destination Factory", ["Thailand", "MM (Myanmar)"])
    with col2:
        req_date_input = st.date_input("Material Required Date", date(2026, 12, 10))
    with col3:
        moq_input = st.number_input("Supplier MOQ (THB)", min_value=0.0, value=moq_threshold)
        
    st.markdown("#### Shipment Breakdown (edit/add rows)")
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
            st.success("Simulation Complete!")
            res1, res2, res3, res4 = st.columns(4)
            res1.metric("True Sourcing Cost", f"THB {results['grand_total']:,.2f}")
            res2.metric("Shipping Cost", f"THB {results['breakdown']['shipping']:,.2f}")
            res3.metric("MOQ Penalties", f"THB {results['breakdown']['moq_excess']:,.2f}")
            res4.metric("Storage Cost", f"THB {results['breakdown']['carrying']:,.2f}")
            
            st.dataframe(pd.DataFrame(results["table"]), use_container_width=True)
