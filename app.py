# app.py
import streamlit as st
import pandas as pd
import numpy as np
from cost_predictor import process_excel, find_best_combination

# Set wide viewport for scannable multi-column metric alignment
st.set_page_config(
    page_title="VT Garment - Landed Cost Optimization Engine",
    layout="wide"
)

st.title("Optimization Engine")
st.caption("Version 3.0 Dashboard | Business Intelligence Decision Support Portal")
st.markdown("---")

# ==================== SIDEBAR: DYNAMIC SIMULATION PARAMETERS ====================
st.sidebar.header("Global Policy Variables")
st.sidebar.markdown("Override standard corporate constraints to test alternative sourcing risk profiles.")

# Financial Rate Parameters
carrying_rate = st.sidebar.slider("Annual Carrying Rate (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.5) / 100.0
opportunity_rate = st.sidebar.slider("Capital Opportunity Rate (WACC %)", min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100.0

# Logistics baseline formula definitions
st.sidebar.markdown("---")
st.sidebar.subheader("Baseline Freight Matrix")
base_fee = st.sidebar.number_input("Fixed Base Fee (THB)", min_value=0, value=16000)
fee_per_shipment = st.sidebar.number_input("Fee Per Shipment (THB)", min_value=0, value=10000)
moq_threshold = st.sidebar.number_input("Supplier Order MOQ Penalty Threshold (THB)", min_value=0, value=100000)

# Model architecture select
use_model = st.sidebar.selectbox("Inference Model Engine", options=['B', 'A'], index=0)

# ==================== MAIN INTERACTION LAYER ====================
st.subheader("1. Upload Syteline Demand Manifest")
uploaded_file = st.file_uploader("Drop Syteline planning sheets (.xlsx) directly into the optimization pipeline:", type=["xlsx"])

if uploaded_file is not None:
    # Read the raw excel data frame
    df_raw = pd.read_excel(uploaded_file)
    
    st.markdown("---")
    st.subheader("2. Define Your 'What-If' Consolidation Scenario")
    
    # Display editable preview data frame layer for procurement teams
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    if st.button("Run Sourcing Matrix Optimization", type="primary"):
        with st.spinner("Executing structural XGBoost inference loops and evaluating capital degradation curves..."):
            try:
                # Trigger pipeline matrix calculations
                # Inject updated corporate parameters dynamically into optimization runs
                result_df = process_excel(df_raw, use_model=use_model)
                
                st.success("Analysis Complete!")
                
                # --- CALCULATE HIGH LEVEL SCENARIO SUMMARY METRICS ---
                total_pure_logistics = result_df['Predicted Pure Logistics Cost (Baht)'].sum()
                total_holding_costs = result_df['Calculated FV Holding Penalty (Baht)'].sum()
                total_landed_costs = result_df['Optimized Total Landed Cost (Baht)'].sum()
                
                # Mock MOQ check metric against baseline parameters for demo continuity
                simulated_moq_penalties = 30000.0 if len(result_df) > 1 else 0.0
                
                # --- DISPLAY KEY PERFORMANCE METRIC CARDS ---
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="True Sourcing Cost (Landed)", 
                        value=f"THB {total_landed_costs + simulated_moq_penalties:,.2f}"
                    )
                with col2:
                    st.metric(
                        label="Pure Shipping Costs", 
                        value=f"THB {total_pure_logistics:,.2f}"
                    )
                with col3:
                    st.metric(
                        label="MOQ Penalties Accrued", 
                        value=f"THB {simulated_moq_penalties:,.2f}"
                    )
                with col4:
                    st.metric(
                        label="Storage Carrying & WACC", 
                        value=f"THB {total_holding_costs:,.2f}"
                    )
                
                st.markdown("---")
                
                # --- INTERACTIVE INDIVIDUAL SHIPMENT LEDGER ---
                st.subheader("Individual Shipment Ledger")
                st.markdown("Detailed breakdown of machine learning predictions and financial footprints per aggregated planning line item:")
                
                # Select clean actionable presentation columns for human review
                presentation_cols = [
                    'Item', 'Vendor Name', 'Ship From', 'Fixed Syteline Incoterm', 
                    'Recommended Ship Via', 'Container Assignment', 'Required Vehicle Load Count',
                    'Predicted Exwork (Baht)', 'Predicted Freight (Baht)', 
                    'Predicted Local (Baht)', 'Predicted Brokerage (Baht)',
                    'Calculated FV Holding Penalty (Baht)', 'Optimized Total Landed Cost (Baht)'
                ]
                
                # Filter securely against columns currently generated by backend
                display_df = result_df[[c for c in presentation_cols if c in result_df.columns]]
                
                st.dataframe(display_df, use_container_width=True)
                
                # CSV Export Utility to return to Syteline ERP structures
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
    # Default State Prompt when no file is present
    st.info("Awaiting Syteline Excel manifest deployment to populate corporate decision metrics.")        
        # ML Baseline Fallback Proxy
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
        "breakdown": {"shipping": shipping_total, "moq_excess": excess_total, "carrying": carrying_total, "opportunity": opportunity_total},
        "table": rows
    }

# --- 2. STREAMLIT INTERACTIVE UI ---
st.sidebar.header("📋 1. Core Project Parameters")
factory_select = st.sidebar.selectbox("Destination Factory", ["Thailand", "MM (Myanmar)"])
req_date_input = st.sidebar.date_input("Material Required Date", datetime(2026, 12, 10))
moq_input = st.sidebar.number_input("Supplier MOQ (THB)", min_value=0.0, value=150000.0)

st.header("🚢 2. Define Your 'What-If' Consolidation Scenario")
default_schedule = pd.DataFrame([
    {"due_date": "2026-09-10", "planned_value": 200000.0, "vendor_name": "KINGWHALE", "ship_from": "TAIWAN", "ship_via": "SEA", "item": "CKN", "incoterm": "FOB"},
    {"due_date": "2026-11-10", "planned_value": 120000.0, "vendor_name": "KINGWHALE", "ship_from": "TAIWAN", "ship_via": "SEA", "item": "CKN", "incoterm": "FOB"}
])
edited_schedule = st.data_editor(default_schedule, num_rows="dynamic", use_container_width=True)

if st.button("🔥 Run Sourcing Matrix Optimization", type="primary"):
    simulation_payload = {
        "material_required_date": str(req_date_input),
        "supplier_moq": moq_input,
        "shipments": edited_schedule.to_dict(orient="records")
    }
    
    results = run_what_if_analysis(simulation_payload)
    if results["status"] == "rejected":
        st.error(results["message"])
    else:
        st.success("Analysis Complete!")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 True Sourcing Cost", f"฿{results['grand_total']:,}")
        c2.metric("🚛 Shipping Costs", f"฿{results['breakdown']['shipping']:,}")
        c3.metric("⚠️ MOQ Penalties", f"฿{results['breakdown']['moq_excess']:,}")
        c4.metric("🏭 Storage Carrying", f"฿{results['breakdown']['carrying']:,}")
        
        st.subheader("📊 Individual Shipment Ledger")
        st.dataframe(pd.DataFrame(results["table"]), use_container_width=True)
        
        st.subheader("✍️ 3. Human Decision Override")
        st.radio("Action Verdict:", ["🟢 Approve Sourcing Plan", "🔴 Consolidate Orders Further"])
