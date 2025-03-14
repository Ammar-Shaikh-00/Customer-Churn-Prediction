# Customer Churn Prediction

## Overview
Customer churn prediction is crucial for businesses to identify customers who are likely to stop using their services. This project uses machine learning to predict customer churn based on various factors such as contract type, payment method, monthly charges, and tenure.

## Features
- **Machine Learning Model**: Trained using customer dataset.
- **Prediction API**: Built with Streamlit for an interactive UI.
- **Dataset**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Model File**: `churn_model.pkl`

## Installation & Setup

### Prerequisites
Make sure you have the following installed:
- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- Streamlit

### Installation Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/Ammar-Shaikh-00/Customer-Churn-Prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
- Upload customer data to predict churn likelihood.
- The model provides a risk assessment of churn probability.
- Businesses can take action based on predictions.

## File Structure
```
├── app.py                   # Streamlit app for user interaction
├── churn_model.pkl          # Trained machine learning model
├── dataset.csv              # Customer dataset
├── Untitled-1.ipynb         # Jupyter Notebook for data exploration
├── README.md                # Project documentation
```

## Contributing
Feel free to fork and enhance the project. Open a pull request if you want to contribute!

## License
This project is open-source under the MIT License.

---
### Author
Muhammad Ammar Shaikh

