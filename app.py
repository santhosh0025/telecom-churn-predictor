import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run 'churn_model.py' first to save the model.")
    st.stop()

# --- Preprocessing Function (Must be the same as in churn_model.py) ---
def preprocess_data(df):
    # Drop customerID column
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop rows with NaN values
    df = df.dropna(subset=['TotalCharges'])

    # Get a list of categorical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Re-align columns with the training data (important for prediction)
    expected_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                     'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                     'MultipleLines_No phone service', 'MultipleLines_Yes',
                     'InternetService_Fiber optic', 'InternetService_No',
                     'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                     'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                     'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                     'TechSupport_No internet service', 'TechSupport_Yes',
                     'StreamingTV_No internet service', 'StreamingTV_Yes',
                     'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                     'Contract_One year', 'Contract_Two year',
                     'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
                     'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    missing_cols = set(expected_cols) - set(df.columns)
    for c in missing_cols:
        df[c] = 0

    return df[expected_cols]

# --- Streamlit App UI ---
st.title('Telecom Customer Churn Predictor')
st.write("Upload a CSV file to predict the churn rate of customers.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Preview of the uploaded data:")
    st.dataframe(df.head())

    # Prediction button
    if st.button('Predict Churn'):
        st.subheader("Prediction Results")
        
        # Make a copy of the uploaded dataframe
        df_for_prediction = df.copy()

        # Drop the 11 rows with empty TotalCharges
        df_for_prediction['TotalCharges'] = pd.to_numeric(df_for_prediction['TotalCharges'], errors='coerce')
        df_for_prediction = df_for_prediction.dropna(subset=['TotalCharges'])

        # Keep the customerID and original Churn for display
        results_df = df_for_prediction[['customerID', 'Churn']].copy()
        
        # Preprocess the data for prediction
        processed_data = preprocess_data(df_for_prediction.drop('Churn', axis=1))

        # Make predictions
        predictions = model.predict(processed_data)

        # Map predictions to 'Yes' or 'No'
        predictions_labels = ['No' if pred == 0 else 'Yes' for pred in predictions]

        # Add predictions to the results DataFrame
        results_df['Predicted_Churn'] = predictions_labels
        
        # Display results
        st.dataframe(results_df)

        # Display counts and a plot
        churn_counts = results_df['Predicted_Churn'].value_counts()
        st.write(f"Predicted number of customers who will churn: {churn_counts.get('Yes', 0)}")
        st.write(f"Predicted number of customers who will not churn: {churn_counts.get('No', 0)}")
        
        fig, ax = plt.subplots()
        sns.countplot(x='Predicted_Churn', data=results_df, ax=ax)
        st.pyplot(fig)
        
    # --- Actionable Insights Section ---
    st.subheader("Actionable Insights")
    st.write("Based on the model, here are some strategies to retain customers.")

    with st.expander("Show Key Factors & Retention Ideas"):
        try:
            # Get feature names and coefficients from the model
            feature_names = model.feature_names_in_
            coefficients = model.coef_[0]
            
            # Create a DataFrame for better display
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).sort_values(by='Coefficient', ascending=False)
            
            st.write("The following features have the strongest positive (increase churn) and negative (decrease churn) influence:")
            st.dataframe(coef_df)

            st.write("---")
        except AttributeError:
            st.write("Note: Coefficient insights are best viewed with a Logistic Regression model.")
            
        st.markdown("#### Top Retention Strategies based on Model Insights:")
        st.markdown("""
        1.  **Target Month-to-month Contracts:** Customers on month-to-month plans are a high churn risk. Offer incentives for them to switch to one or two-year contracts.
        
        2.  **Improve Technical Support:** Customers without tech support are more likely to churn. Offer free or low-cost tech support as an added-value service.
        
        3.  **Encourage Digital Billing & Payment:** The model often finds a link between paperless billing and electronic checks and churn. Encourage customers to use more stable payment methods and simplify the billing experience.
        
        4.  **Reward Long-Term Customers:** Customers with higher `tenure` are less likely to churn. Create loyalty programs, special offers, and discounts for long-term customers to ensure they feel valued.
        
        5.  **Review Service Pricing:** Customers with high `MonthlyCharges` tend to churn more. Analyze pricing tiers and offer personalized discounts to customers at risk.
        """)