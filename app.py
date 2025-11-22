import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Customer Churn System", layout="wide")

# ✅ ADD DARK BLUE BACKGROUND + BLACK UPLOADER
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #001F3F;
}
[data-testid="stSidebar"] {
    background-color: #001A33;
}
h1, h2, h3, h4, h5, p, label, span {
    color: white !important;
}
/* ===== FILE UPLOADER DARK ===== */
[data-testid="stFileUploader"] section div {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #444444 !important;
    border-radius: 5px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- HEADER & AIM SECTION ----------------
st.markdown("""
<h1 style='color:#4E9AFF;'>📊 Customer Churn Prediction & Insights</h1>

### 🎯 Aim of This Website

This system helps businesses to:

✅ Predict which customers are likely to leave  
✅ Understand *why* a customer may churn  
✅ Take proactive retention actions  
✅ Reduce revenue loss and improve loyalty  

---
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict Churn", "Analytics"])

# =====================================================
# ====================== HOME =========================
# =====================================================

if page == "Home":
    st.markdown("""
    ## 👋 Welcome!

    Upload your customer dataset and explore:

    ✅ Auto-generated customer form  
    ✅ One-click churn prediction  
    ✅ Smart retention suggestions  
    ✅ Visual insights & patterns  

    ---
    """)

# =====================================================
# ===================== PREDICT =======================
# =====================================================

elif page == "Predict Churn":

    uploaded_file = st.file_uploader("📌 Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")

        st.write("### 🔍 Preview of Data")
        st.dataframe(df.head())

        st.write("### 📝 Auto-Generated Customer Input Form")

        user_input = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                user_input[col] = st.selectbox(f"{col}", df[col].astype(str).unique())
            else:
                user_input[col] = st.number_input(
                    f"{col}",
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=float(df[col].mean())
                )

        input_df = pd.DataFrame([user_input])
        st.markdown("---")

        try:
            model = pickle.load(open("model.pkl", "rb"))
        except:
            st.error("❗ Model not found! Please add model.pkl to your folder.")
            st.stop()

        st.markdown("### 🤖 Predict & Recommend")

        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = input_df[col].astype('category').cat.codes

        required_features = model.feature_names_in_
        input_df = input_df.reindex(columns=required_features, fill_value=0)

        if st.button("🔮 Predict Churn"):
            prediction = model.predict(input_df)[0]

            if hasattr(model, "feature_importances_"):
                feature_scores = pd.DataFrame({
                    "Feature": required_features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                top_factors = feature_scores.head(3)
                st.write("### 🧠 Why This Prediction?")
                st.table(top_factors)

            if prediction == 1:
                st.markdown("<h2 style='color:#FF4B4B;'>⚠️ High Churn Risk Detected</h2>", unsafe_allow_html=True)
                st.subheader("✅ Personalized Retention Strategies")
                st.write("""
                - Offer personalized discounts or loyalty rewards  
                - Reduce waiting time & improve support follow-ups  
                - Provide onboarding & product education  
                - Offer flexible billing or temporary downgrade  
                """)
                st.info("""
                *“We value your journey with us. Here's something special crafted just for you!”*
                """)
            else:
                st.markdown("<h2 style='color:#00FF8C;'>✅ Customer Likely to Stay</h2>", unsafe_allow_html=True)
                st.subheader("🎉 Retention Boost Ideas")
                st.markdown("""
                <ul style='color:white;'>
                    <li>Send appreciation & reward points</li>
                    <li>Share new features and exclusive offers</li>
                    <li>Collect positive feedback</li>
                    <li>Celebrate usage milestones</li>
                </ul>
                """, unsafe_allow_html=True)

    else:
        st.warning("📂 Please upload a CSV file to continue.")

# =====================================================
# ==================== ANALYTICS ======================
# =====================================================

elif page == "Analytics":

    uploaded_file = st.file_uploader("📌 Upload dataset for analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded!")

        st.write("### 📊 Churn Distribution")
        if "Churn" in df.columns:
            fig = px.pie(df, names="Churn", title="Churn vs Retained")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'Churn' not found — cannot create churn chart.")

        st.markdown("---")

        st.write("### 📈 Numerical Feature Trends")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select a column to visualize:", numeric_cols)
            fig2 = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No numeric columns found for visualization.")
