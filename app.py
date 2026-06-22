import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="VTG Sourcing Optimizer")
st.title("👔 VT Garment Co., Ltd. - Sourcing Optimization Engine")
st.caption("Version 3.0 Dashboard | Business Intelligence Decision Support Portal")

# --- 1. CORE METHODOLOGY ENGINE ---
C_RATE = 0.06  # 6% p.a. Physical Inventory Carrying Cost
O_RATE = 0.10  # 10% p.a. Capital Opportunity Cost

def run_what_if_analysis(payload):
    try:
        req_date = datetime.strptime(payload["material_required_date"], "%Y-%m-%d")
    except Exception:
        return {"status": "error", "message": "Invalid Material Required Date format."}
        
    moq = float(payload.get("supplier_moq", 0))
    shipping_total, excess_total, carrying_total, opportunity_total = 0.0, 0.0, 0.0, 0.0
    rows = []
    
    for i, ship in enumerate(payload.get("shipments", [])):
        due_date = datetime.strptime(ship["due_date"], "%Y-%m-%d")
        days_early = (req_date - due_date).days
        
        if days_early < 0:
            return {
                "status": "rejected",
                "message": f"❌ Operational Failure: Shipment #{i+1} arriving on {ship['due_date']} is LATE!"
            }
            
        val = float(ship["planned_value"])
        excess = max(0.0, moq - val)
        effective_val = max(val, moq)
        
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
