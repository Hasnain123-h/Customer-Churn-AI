    
# import streamlit as st
# import pandas as pd
# import requests
# import joblib
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent.parent
# PREPROCESSOR_PATH = (
#     BASE_DIR / "models" / "preprocessor.pkl"
# )
# MODEL_PATH = (
#     BASE_DIR / "models" / "logistic_regression_tuned.pkl"
# )
# preprocessor = joblib.load(PREPROCESSOR_PATH)
# model = joblib.load(MODEL_PATH)


from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "rag"
sys.path.insert(0, str(RAG_DIR))
from QA import answer_question


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


API_URL = "http://127.0.0.1:8000/predict"

HISTORY_URL = "http://127.0.0.1:8000/predictions"


st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #1f2937,
            #374151
        );
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        margin: 0;
        color: #e5e7eb;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 15px;
        color: #6b7280;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Churn Prediction</h1>
        <p>
            AI-Powered Customer Retention Prediction System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="section-title">
        Customer Information
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter the customer's information below to predict "
    "the probability of churn."
)


with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col2:
        st.subheader("Services")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        st.subheader("Subscription")

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year" ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    st.divider()
    st.subheader("Billing Information")
    charge_col1, charge_col2 = st.columns(2)

    with charge_col1:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with charge_col2:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=850.0,
            step=10.0
        )

    st.write("")
    submit_button = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

#prediction
if submit_button:
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    customer_data = input_data.iloc[0].to_dict()

    try:
        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=30
        )

        response.raise_for_status()
        result_data = response.json()
        prediction = result_data["prediction"]
        result = result_data["result"]
        probability_percent = result_data[
            "churn_probability"
        ]


        st.divider()
        st.markdown(
            """
            <div class="section-title">
                Prediction Result
            </div>
            """,
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if prediction == 1:
                st.error(
                    "Customer is likely to Churn"
                )

            else:
                st.success(
                    "Customer is likely to Stay"
                )

        with result_col2:
            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )


        st.write("Churn Risk Level")
        st.progress(
            min(
                max(
                    int(probability_percent),
                    0
                ),
                100
            )
        )


        if probability_percent >= 70:
            st.warning(
                "High churn probability. "
                "The customer may require retention attention."
            )

        elif probability_percent >= 40:
            st.info(
                "Moderate churn probability. "
                "The customer may need monitoring."
            )

        else:
            st.success(
                "Low churn probability. "
                "The customer is currently less likely to churn."
            )


        st.session_state["customer_data"] = customer_data
        st.session_state["prediction"] = prediction
        st.session_state["result"] = result
        st.session_state[
            "probability_percent"
        ] = probability_percent


    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI se connection nahi ho paya. "
            "Please make sure FastAPI is running."
        )


    except requests.exceptions.Timeout:
        st.error(
            "API request timeout ho gayi. "
            "Please try again."
        )


    except requests.exceptions.HTTPError as error:
        st.error(
            f"FastAPI error: {error}"
        )


    except requests.exceptions.RequestException as error:
        st.error(
            f"API request failed: {error}"
        )


    except Exception as error:
        st.error(
            f"Unexpected error: {error}"
        )


st.divider()
st.markdown(
    """
    <div class="section-title">
        🤖 AI Retention Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Ask the AI assistant about customer churn "
    "and get a personalized retention recommendation."
)


question = st.text_input(
    "Ask your question:",
    placeholder="Why is this customer at risk of churn?"
)

if st.button(
    "Ask AI",
    use_container_width=True
):

    if "customer_data" not in st.session_state:
        st.warning(
            "Please predict customer churn first."
        )

    elif not question.strip():
        st.warning(
            "Please enter a question."
        )

    else:        
        customer_info = {
            **st.session_state["customer_data"],
            "prediction": st.session_state[
                "prediction"
            ],
            "result": st.session_state[
                "result"
            ],
            "churn_probability": st.session_state[
                "probability_percent"
            ]
        }

        try:
            with st.spinner("AI is analyzing the customer..."):
                ai_answer = answer_question(
                    question,
                    customer_info
                )


            st.markdown(
                "### AI Answer"
            )

            st.write(
                ai_answer
            )


        except Exception as error:
            st.error(
                f"AI Assistant Error: {error}"
            )


st.divider()
st.markdown(
    """
    <div class="section-title">
        Prediction History
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "View previous customer churn predictions "
    "saved in the database."
)


if st.button(
    "View Prediction History",
    use_container_width=True
):

    try:
        response = requests.get(
            HISTORY_URL,
            timeout=30
        )

        response.raise_for_status()
        history_data = response.json()
        predictions = history_data.get(
            "predictions",
            []
        )


        if predictions:
            df_history = pd.DataFrame(
                predictions
            )

            st.dataframe(
                df_history,
                use_container_width=True
            )

        else:
            st.info(
                "No prediction history found."
            )


    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI se connection nahi ho paya. "
            "Please make sure FastAPI is running."
        )


    except requests.exceptions.Timeout:
        st.error(
            "Prediction history request timeout ho gayi."
        )


    except requests.exceptions.HTTPError as error:
        st.error(
            f"FastAPI error: {error}"
        )


    except requests.exceptions.RequestException as error:
        st.error(
            f"API request failed: {error}"
        )
    except Exception as error:

        st.error(
            f"Unexpected error: {error}"
        )


st.markdown(
    """
    <div class="footer">
        Customer Churn Prediction |
        Machine Learning + RAG + LLM Project
    </div>
    """,
    unsafe_allow_html=True
)