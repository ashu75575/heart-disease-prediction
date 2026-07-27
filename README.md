# Heart Disease Prediction App

A machine learning web application that predicts the risk of a heart stroke based on a patient's medical and demographic information. 

This project explores various classification algorithms (Random Forest, XGBoost, Naive Bayes, etc.) and deploys the best-performing model using **Streamlit** for an interactive, user-friendly frontend.

## Features

- **Interactive UI**: Built with Streamlit, allowing users to manually input medical features.
- **Accurate Predictions**: Uses a fine-tuned XGBoost machine learning model.
- **Data Scaling**: Automatically scales numerical data and one-hot encodes categorical data under the hood.
- **Testing Script**: Includes a standalone `test_app.py` script to run multiple data points and verify prediction pipeline logic.

## Project Structure

- `app.py`: The main Streamlit web application.
- `Heartdisease.ipynb`: Jupyter Notebook containing data exploration, preprocessing, model training, and hyperparameter tuning.
- `test_app.py`: A Python script simulating user inputs to test the model logic without launching the web app.
- `heart.csv`: The dataset used to train the machine learning models.
- `requirements.txt`: Python dependencies required to run the project.
- **Models & Assets**:
  - `XGBoost-tuned-model.pkl`: The tuned XGBoost model currently serving predictions.
  - `scaler.pkl`: The StandardScaler object to scale numerical features.
  - `columns.pkl`: Contains the exact expected feature columns post one-hot-encoding.

## Setup & Installation

### 1. Create a Virtual Environment

It is highly recommended to isolate dependencies using a virtual environment:

```bash
# Create the environment
python3 -m venv venv

# Activate the environment (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the Web App

Once the environment is active and dependencies are installed, launch the Streamlit server:

```bash
streamlit run app.py
```

The app will open automatically in your browser (usually at `http://localhost:8501`).

### 4. Run the Test Script

To verify that the model logic works via the command line on multiple test cases without running the UI, execute:

```bash
python test_app.py
```
