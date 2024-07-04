
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objs as go

# Set page config
st.set_page_config(page_title="Mental Health Disorder Classification", layout="wide")

# Main title with custom style
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Mental Health Disorder Classification</h1>", unsafe_allow_html=True)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

# Load the dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df = load_data('mental_health_dataset.csv')

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply text preprocessing to 'Symptoms'
df['Symptoms'] = df['Symptoms'].apply(preprocess_text)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

@st.cache_resource
def train_and_evaluate_for_disorder(disorder, model_name, _model, random_state=None):
    # Create a binary classification problem for the specific disorder
    df_disorder = df.copy()
    df_disorder['Target'] = (df_disorder['Disorder'] == disorder).astype(int)
    
    # Prepare the data
    X = df_disorder['Symptoms']
    y = df_disorder['Target']
    
    # Create a pipeline with vectorizer and scaler
    preprocess_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    # Perform stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []
    
    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Preprocess the data
        X_train_preprocessed = preprocess_pipeline.fit_transform(X_train)
        X_val_preprocessed = preprocess_pipeline.transform(X_val)
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
        
        # Train the model
        model = _model.__class__(**_model.get_params())  # Create a new instance of the model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict and calculate balanced accuracy
        y_pred = model.predict(X_val_preprocessed)
        cv_scores.append(balanced_accuracy_score(y_val, y_pred))
    
    # Train on the full dataset for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    
    # Preprocess the data
    X_train_preprocessed = preprocess_pipeline.fit_transform(X_train)
    X_test_preprocessed = preprocess_pipeline.transform(X_test)
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    
    # Train the model
    model = _model.__class__(**_model.get_params())  # Create a new instance of the model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict
    y_pred = model.predict(X_test_preprocessed)
    y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]
    
    # Calculate multiple metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'cv_balanced_accuracy_mean': np.mean(cv_scores),
        'cv_balanced_accuracy_std': np.std(cv_scores),
        'test_accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'confusion_matrix': conf_matrix
    }

# Sidebar controls
st.sidebar.header('Model Training')
disorder = st.sidebar.selectbox('Select Disorder', df['Disorder'].unique())
selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

# Train model button
if st.sidebar.button('Train Model'):
    with st.spinner(f"Training {selected_model} for {disorder}..."):
        results = train_and_evaluate_for_disorder(disorder, selected_model, models[selected_model], random_state=42)
    
    st.success(f"Trained {selected_model} for {disorder}.")
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Balanced Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC'],
        'Value': [
            results['balanced_accuracy'],
            results['f1_score'],
            results['precision'],
            results['recall'],
            results['auc_roc']
        ]
    })
    st.table(metrics_df.set_index('Metric'))
    
    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    conf_matrix = results['confusion_matrix']
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 20},
    ))
    fig.update_layout(
        title=f'Confusion Matrix: {selected_model} - {disorder}',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    st.plotly_chart(fig)

# Train all models button
if st.sidebar.button("Train All Models"):
    results = {}
    progress_bar = st.progress(0)
    for i, (model_name, model) in enumerate(models.items()):
        model_results = {}
        for j, disorder in enumerate(df['Disorder'].unique()):
            with st.spinner(f"Training {model_name} for {disorder}..."):
                disorder_results = train_and_evaluate_for_disorder(disorder, model_name, model, random_state=42)
                model_results[disorder] = disorder_results
            progress = (i * len(df['Disorder'].unique()) + j + 1) / (len(models) * len(df['Disorder'].unique()))
            progress_bar.progress(progress)
        results[model_name] = model_results
    
    st.success("All models have been trained!")
    
    # Create comparison plots
    st.subheader("Model Comparison")
    metrics_to_plot = ['balanced_accuracy', 'f1_score', 'auc_roc']
    for metric in metrics_to_plot:
        data = []
        for model, model_results in results.items():
            for disorder, disorder_results in model_results.items():
                data.append({
                    'Model': model,
                    'Disorder': disorder,
                    metric: disorder_results[metric]
                })
        df_plot = pd.DataFrame(data)
        
        fig = px.bar(df_plot, x='Disorder', y=metric, color='Model', barmode='group',
                     title=f'{metric.replace("_", " ").title()} Comparison Across Models and Disorders')
        fig.update_layout(xaxis_title='Disorder', yaxis_title=metric.replace("_", " ").title())
        st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Other Options")
if st.sidebar.button("Clear All Models"):
    st.caching.clear_cache()
    st.sidebar.success("All models have been cleared.")
    st.experimental_rerun()
