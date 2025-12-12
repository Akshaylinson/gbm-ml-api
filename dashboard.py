import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="ML API Dashboard", layout="wide")

API_BASE = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer demo-key-123"}

def get_metrics():
    try:
        response = requests.get(f"{API_BASE}/metrics", headers=HEADERS)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_health():
    try:
        response = requests.get(f"{API_BASE}/health")
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

st.title("🤖 ML API Monitoring Dashboard")

# Health Status
col1, col2, col3, col4 = st.columns(4)

health = get_health()
with col1:
    status = health.get("status", "unknown")
    st.metric("API Status", status.upper(), delta="Healthy" if status == "healthy" else "Down")

with col2:
    uptime = health.get("uptime_seconds", 0)
    st.metric("Uptime", f"{uptime/3600:.1f}h")

# Metrics
metrics = get_metrics()

with col3:
    total_requests = metrics.get("total_requests", 0)
    st.metric("Total Requests", total_requests)

with col4:
    error_rate = metrics.get("error_rate", 0)
    st.metric("Error Rate", f"{error_rate:.2%}")

# Performance Metrics
st.subheader("📊 Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    avg_response_time = metrics.get("avg_response_time_ms", 0)
    st.metric("Avg Response Time", f"{avg_response_time:.2f}ms")

with col2:
    avg_probability = metrics.get("avg_probability", 0)
    st.metric("Avg Prediction Probability", f"{avg_probability:.3f}")

# Prediction Distribution
if "prediction_distribution" in metrics:
    dist = metrics["prediction_distribution"]
    
    fig = go.Figure(data=[
        go.Bar(name='Negative', x=['Predictions'], y=[dist.get("negative", 0)]),
        go.Bar(name='Positive', x=['Predictions'], y=[dist.get("positive", 0)])
    ])
    fig.update_layout(title="Prediction Distribution (Last Hour)")
    st.plotly_chart(fig, use_container_width=True)

# Feature Drift Detection
if "feature_drift" in metrics:
    drift_data = metrics["feature_drift"]
    if drift_data:
        st.subheader("🚨 Feature Drift Detection")
        
        drift_df = pd.DataFrame([
            {"Feature": k, "Drift Score": v} 
            for k, v in drift_data.items()
        ])
        
        fig = px.bar(drift_df, x="Feature", y="Drift Score", 
                    title="Feature Drift Scores (Higher = More Drift)")
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                     annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True)

# Test API
st.subheader("🧪 Test API")

with st.form("test_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Income", min_value=-50000, max_value=500000, value=55000)
        balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=1200)
    
    with col2:
        city = st.selectbox("City", ["A", "B", "C"])
        has_credit_card = st.selectbox("Has Credit Card", [0, 1])
        explain = st.checkbox("Include Explanation")
    
    if st.form_submit_button("Make Prediction"):
        payload = {
            "rows": [{
                "age": age,
                "income": income,
                "balance": balance,
                "city": city,
                "has_credit_card": has_credit_card
            }],
            "explain": explain
        }
        
        try:
            response = requests.post(f"{API_BASE}/predict", json=payload, headers=HEADERS)
            if response.status_code == 200:
                result = response.json()
                prediction = result["results"][0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", prediction["prediction"])
                with col2:
                    st.metric("Probability", f"{prediction['probability']:.3f}")
                with col3:
                    st.metric("Risk Score", prediction.get("risk_score", "N/A"))
                
                if prediction.get("explanation"):
                    st.subheader("Feature Importance")
                    exp_df = pd.DataFrame([
                        {"Feature": k, "Importance": v} 
                        for k, v in prediction["explanation"].items()
                    ])
                    fig = px.bar(exp_df, x="Feature", y="Importance")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# Auto-refresh
if st.button("🔄 Refresh Dashboard"):
    st.experimental_rerun()