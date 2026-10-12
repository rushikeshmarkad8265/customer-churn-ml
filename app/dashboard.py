import streamlit as st
import requests

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn & Retention Intelligence")
st.write("Estimate customer churn risk using the trained ML model.")

with st.form("customer_form"):
    st.subheader("Customer Information")

    col1,col2 = st.columns(2)

    with col1:
        age = st.number_input("age",min_value=18,max_value=100,value=35)
        gender = st.selectbox("Gender",["Male","Female"])
        tenure = st.number_input("tenure",min_value=1,value=24)
        usage_frequency = st.number_input("usage_frequency",
                                          min_value=0,value=24)
        support_calls = st.number_input("support_calls",min_value=0,value=3)

    with col2:
        payment_delay = st.number_input("payment delay",min_value=0,value=5)

        subscription_type = st.selectbox("Subscription Type",["Basic","Standard","Premium"])

        contract_length = st.selectbox("Contract",["Monthly", "Quarterly", "Annual"])

        total_spend = st.number_input("total spend", min_value=0.00,value=600.00)

        last_interaction = st.number_input("Last interaction",min_value=0,value=10)

    submitted = st.form_submit_button("Predict the churn")


if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "subscription_type": subscription_type,
        "contract_length": contract_length,
        "total_spend": total_spend,
        "last_interaction": last_interaction,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Results")

        probability = result["churn_probability"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )
        reasons = result.get("top_reasons", [])



        col2.metric(
            "Predicted Class",
            "Churn" if result["churn_prediction"] == 1
            else "No Churn"
        )

        col3.metric("Risk Level", result["risk_level"])
        if reasons:
            st.subheader("Why did the model make this prediction?")

            for reason in reasons:
                direction = reason["direction"]
                feature = reason["feature"]
                value = reason["shap_value"]

                if direction == "toward_churn":
                    explanation = "Pushes toward churn"
                elif direction == "away_from_churn":
                    explanation = "Pushes away from churn"
                else:
                    explanation = "Neutral contribution"

                st.write(f"**{feature}** — {explanation}")
                st.caption(f"SHAP contribution: {value:+.4f}")


        st.progress(probability)
        st.caption(result.get("note", ""))

    except requests.RequestException as exc:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure FastAPI is running at http://127.0.0.1:8000."
        )
        st.caption(str(exc))


