# AI-in-Medical-HealthCare
# Mental Health Disorder Classification

This project aims to classify mental health disorders based on symptom descriptions using machine learning models. It includes a Streamlit-based web application that enables users to train and evaluate machine learning models for different disorders, visualize performance metrics, and compare models.

## Project Overview

The main objective of this application is to assist healthcare professionals and researchers by providing a tool that predicts mental health disorders based on symptoms. The application uses multiple models, including Random Forest and Logistic Regression, and employs techniques such as SMOTE for handling class imbalance, making it suitable for imbalanced datasets often found in medical data.

## Features

- **Model Selection**: Choose from different models, including Random Forest and Logistic Regression, for training on specific disorders.
- **Performance Metrics**: Evaluate models with metrics such as balanced accuracy, F1 score, precision, recall, and AUC-ROC.
- **Confusion Matrix Visualization**: Plot and analyze confusion matrices to understand model performance better.
- **Model Comparison**: Train all models across disorders and compare their performance through interactive visualizations.
- **Clear Cache**: Clear model training data from cache with one click.

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed along with the following dependencies:

- Streamlit
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- plotly

You can install the necessary packages with:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/harshchi19/AI-in-Medical-HealthCare-.git
   cd AI-in-Medical-HealthCare-
   ```

2. Place your dataset (`mental_health_dataset.csv`) in the project root.

3. Run the application:

   ```bash
   streamlit run main.py
   ```

## File Structure

- **main.py**: The main Streamlit application file.
- **requirements.txt**: List of dependencies required to run the application.
- **mental_health_dataset.csv**: Input data file containing symptoms and disorder labels (ensure this file is in the project root).

## Usage

1. **Upload Dataset**: Load the `mental_health_dataset.csv` file. Ensure it contains fields for symptoms and disorder labels.
2. **Model Training**: Choose a disorder and model, then train it using the sidebar options. Metrics are displayed after training.
3. **Model Comparison**: Train all models across all disorders and visualize performance comparisons.
4. **Clear Cache**: Use the "Clear All Models" button to reset the cached data.

## Dataset

The application requires a CSV file named `mental_health_dataset.csv` with the following fields:

- **Symptoms**: Descriptions of symptoms in text form.
- **Disorder**: Label for the corresponding mental health disorder.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any additions or changes.

## License

This project is open source and available under the MIT License.
