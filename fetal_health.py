import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 
# Display gif
st.image('fetal_health_image.gif', width = True)

st.write("Utalize the advanced machine learning application to predict fetal health classification.")

# Load the pre-trained models from the pickle file
# Decision Tree:
dt_pickle = open('decision_tree_fetal_health.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
dt_pickle.close()

# Random Forest:
rf_pickle = open('random_forest_fetal_health.pickle', 'rb') 
model_cv = pickle.load(rf_pickle) 
rf_pickle.close()

# AdaBoost:
ada_pickle = open('adaboost_fetal_health.pickle', 'rb') 
grid_cv = pickle.load(ada_pickle) 
ada_pickle.close()

# Soft Voting:
svc_pickle = open('soft_voting_fetal_health.pickle', 'rb') 
svc_clf = pickle.load(svc_pickle) 
svc_pickle.close()

# Load the default dataset
default_df = pd.read_csv('fetal_health.csv')

##### -------------------   SIDEBAR   ---------------------- ######
# Create a sidebar for input collection
st.sidebar.header('Fetal Health Features Input')

# Option 1: Asking users to input their data as a file
file = st.sidebar.file_uploader("Upload Your Data", type=["csv"])
st.sidebar.warning('Ensure your data strictly follows the format outlined below.')

st.sidebar.write(default_df.head(5))

#---------------------------------------------------------------------------------------------
# Using Default (Original) Dataset to Automate Few Items
#---------------------------------------------------------------------------------------------

default_df = pd.read_csv('fetal_health.csv')
default_df.dropna(inplace = True)

model = st.sidebar.radio('Choose Model for Prediction', options = ['Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'])

if file is None:
    st.info("Please upload data to be processed")

else:
    st.success("CSV file uploaded successfully", icon = "✅")
    # Loading data
    user_df = pd.read_csv(file) # User provided data
    original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model

    # Dropping null values
    user_df = user_df.dropna() 
    original_df = original_df.dropna() 

    # Remove output columns from original data
    original_df = original_df.drop(columns = ['fetal_health'])

    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    if model == 'Decision Tree':
        st.sidebar.success('You selected Decision Tree')
        # Using predict() with new data provided by the user

        # Predictions for user data
        user_pred = clf.predict(user_df_encoded)
        user_pred_prob = clf.predict_proba(user_df_encoded)

        # Adding predicted species to user dataframe
        user_df['Predicted Fetal Health'] = user_pred

        # Changing row names
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map({ 
            1: 'Normal', 2: 'Suspect', 3: 'Pathological'})
        
        # Adding prediction probability percentage to user dataframe 
        user_df['Prediction Probability %'] = user_pred_prob.max(axis=1) * 100

        # Function to apply color 
        def color_fetal_health(val): 
            if val == 'Normal': color = 'lime' 
            elif val == 'Suspect': color = 'yellow' 
            elif val == 'Pathological': color = 'orange' 
            return f'background-color: {color}'
        
        # Applying color to the dataframe 
        user_df_styled = user_df.style.applymap(color_fetal_health, subset=['Predicted Fetal Health']) 
        user_df_styled = user_df_styled.format({"Prediction Probability %": "{:.2f}"})
        
        # Show the predicted species on the app 
        st.subheader("Predicting Fetal Health Class Using Decision Tree Model")
        st.dataframe(user_df_styled)

        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Visualizing Decision Tree
        with tab1:
            st.write("### Decision Tree Visualization")
            st.image('dt_visual.svg')
            st.caption("Visualization of the Decision Tree used in prediction.")

        # Tab 2: Feature Importance Visualization
        with tab2:
            st.write("### Feature Importance")
            st.image('dt_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab3:
            st.write("### Confusion Matrix")
            st.image('dt_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab4:
            st.write("### Classification Report")
            report_df = pd.read_csv('dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each.")
    
    elif model == 'Random Forest':
        st.sidebar.success('You selected Random Forest')
        # Using predict() with new data provided by the user

        # Predictions for user data
        user_pred = model_cv.predict(user_df_encoded)
        user_pred_prob = model_cv.predict_proba(user_df_encoded)

        # Adding predicted species to user dataframe
        user_df['Predicted Fetal Health'] = user_pred

        # Changing row names
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map({ 
            1: 'Normal', 2: 'Suspect', 3: 'Pathological'})
        
        # Adding prediction probability percentage to user dataframe 
        user_df['Prediction Probability %'] = user_pred_prob.max(axis=1) * 100

        # Function to apply color 
        def color_fetal_health(val): 
            if val == 'Normal': color = 'lime' 
            elif val == 'Suspect': color = 'yellow' 
            elif val == 'Pathological': color = 'orange' 
            return f'background-color: {color}'
        
        # Applying color to the dataframe 
        user_df_styled = user_df.style.applymap(color_fetal_health, subset=['Predicted Fetal Health']) 
        user_df_styled = user_df_styled.format({"Prediction Probability %": "{:.2f}"})
        
        # Show the predicted species on the app 
        st.subheader("Predicting Fetal Health Class Using Random Forest Model")
        st.dataframe(user_df_styled)

        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 2: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('rf_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('rf_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('rf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each.")

    elif model == 'AdaBoost':
        st.sidebar.success('You selected AdaBoost')
        # Using predict() with new data provided by the user

        # Predictions for user data
        user_pred = clf.predict(user_df_encoded)
        user_pred_prob = clf.predict_proba(user_df_encoded)

        # Adding predicted species to user dataframe
        user_df['Predicted Fetal Health'] = user_pred

        # Changing row names
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map({ 
            1: 'Normal', 2: 'Suspect', 3: 'Pathological'})
        
        # Adding prediction probability percentage to user dataframe 
        user_df['Prediction Probability %'] = user_pred_prob.max(axis=1) * 100

        # Function to apply color 
        def color_fetal_health(val): 
            if val == 'Normal': color = 'lime' 
            elif val == 'Suspect': color = 'yellow' 
            elif val == 'Pathological': color = 'orange' 
            return f'background-color: {color}'
        
        # Applying color to the dataframe 
        user_df_styled = user_df.style.applymap(color_fetal_health, subset=['Predicted Fetal Health']) 
        user_df_styled = user_df_styled.format({"Prediction Probability %": "{:.2f}"})
        
        # Show the predicted species on the app 
        st.subheader("Predicting Fetal Health Class Using AdaBoost Model")
        st.dataframe(user_df_styled)

        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 2: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('dt_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('dt_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each.")
    
    elif model == 'Soft Voting':
        st.sidebar.success('You selected Soft Voting')
        # Using predict() with new data provided by the user

        # Predictions for user data
        user_pred = svc_clf.predict(user_df_encoded)
        user_pred_prob = svc_clf.predict_proba(user_df_encoded)

        # Adding predicted species to user dataframe
        user_df['Predicted Fetal Health'] = user_pred

        # Changing row names
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map({ 
            1: 'Normal', 2: 'Suspect', 3: 'Pathological'})
        
        # Adding prediction probability percentage to user dataframe 
        user_df['Prediction Probability %'] = user_pred_prob.max(axis=1) * 100

        # Function to apply color 
        def color_fetal_health(val): 
            if val == 'Normal': color = 'lime' 
            elif val == 'Suspect': color = 'yellow' 
            elif val == 'Pathological': color = 'orange' 
            return f'background-color: {color}'
        
        # Applying color to the dataframe 
        user_df_styled = user_df.style.applymap(color_fetal_health, subset=['Predicted Fetal Health']) 
        user_df_styled = user_df_styled.format({"Prediction Probability %": "{:.2f}"})
        
        # Show the predicted species on the app 
        st.subheader("Predicting Fetal Health Class Using Soft Voting Model")
        st.dataframe(user_df_styled)

        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 2: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('dt_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('dt_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each.")