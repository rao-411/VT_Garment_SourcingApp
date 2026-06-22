# VT Garment Co., Ltd. — Sourcing Optimization Engine
An interactive decision-support system developed for the VT Garment Case Competition to optimize raw material procurement schedules.

## Project Overview
This application addresses the classic logistics trade-off: balancing freight consolidation against inventory holding penalties. The platform couples a machine learning predictive model (trained on historical shipping manifests) with a time-value-of-money financial simulation layer based on the company's Version 3.0 Sourcing Cost Methodology.

By allowing procurement managers to run multi-shipment "What-If" scenarios, the app quantifies hidden inventory holding costs, identifies supplier Minimum Order Quantity (MOQ) penalties, and prevents production stockouts.

## Core Features
* **Predictive Logistics Infrastructure:** A dual-stage XGBoost Regressor architecture that predicts component-level shipping overhead (Ex-Works, Freight, Local Charges, and Brokerage) based on vendor, trade lane, and material volume parameters.
* **Incoterm Business Logic:** Embedded rule structures that programmatically filter out shipping cost components based on commercial definitions (e.g., forcing logistics legs to zero for DDP terms).
* **Dynamic Scenario Simulation:** A free-form grid interface where users can split an order into any configuration of delivery dates and shipment values to compare cost thresholds live.
* **Financial Risk Quantification:** Automated integration of approved FY2026 corporate capital metrics—specifically a 6% p.a. Warehouse Carrying Cost and a 10% p.a. Opportunity Cost of Capital.

## Repository Structure
* `app.py`: Main Streamlit application file handling user interface interactions and scenario calculation routines.
* `cost_predictor.py`: Backend execution script that manages data formatting, mapping, and model inference loops.
* `model.ipynb`: Data science development notebook detailing historical data exploration, cleaning, log transformations, and XGBoost training cycles.
* `requirements.txt`: System dependencies specification for automated cloud environment deployment.
* `*.pkl / *.json`: Saved model pipelines, weights, metadata, and structural lookup variables.

## Getting Started

### Local Setup
1. Clone this repository.
2. Ensure you have the required packages installed:
   ```bash
   pip install -r requirements.txt
