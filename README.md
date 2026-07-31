# Assignment-10
# ❤️ Heart Disease Prediction using Machine Learning and Flask API

## 📌 Objective

The objective of this project is to develop a machine learning model that predicts whether a patient is at risk of heart disease based on clinical parameters. A **Random Forest Classifier** is trained using the Heart Disease dataset, saved using **Joblib**, and integrated into a **Flask REST API** that accepts patient data in JSON format and returns the prediction.

---

## 📂 Dataset Link

**Dataset Name:** Heart Disease Prediction Dataset

**Source:** https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

---

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- KaggleHub

---

## ⚙️ Methodology

1. Loaded the Heart Disease dataset using **KaggleHub**.
2. Displayed the first five records and dataset information.
3. Identified numerical features and the target variable.
4. Checked for missing values.
5. Split the dataset into **80% training** and **20% testing**.
6. Trained a **Random Forest Classifier**.
7. Evaluated the model using **Accuracy Score**.
8. Saved the trained model using **Joblib** (`model.pkl`).
9. Developed a **Flask REST API** to load the trained model and return predictions in JSON format.

---

## 🤖 Machine Learning Model

**Algorithm Used:** Random Forest Classifier

### Model Workflow

- Load dataset
- Data preprocessing
- Train-test split (80:20)
- Train Random Forest model
- Evaluate using Accuracy Score
- Save model using Joblib
- Load model in Flask API
- Predict heart disease risk from JSON input

---

## 📊 Results

| Metric | Score |
|---------|------:|
| **Accuracy** | **Add your result** |

### Example API Request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Example API Response

```json
{
  "prediction": "Heart Disease Detected"
}
```

---

## 📁 Project Structure

```
HeartDiseaseDeployment/
│── app.py
│── train_model.py
│── model.pkl
│── requirements.txt
│── README.md
│── heart.csv
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/HeartDiseaseDeployment.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This generates the trained model:

```
model.pkl
```

### 4. Run the Flask API

```bash
python app.py
```

The API will start locally at:

```
http://127.0.0.1:5000/
```

### 5. Test the Prediction Endpoint

**POST**

```
http://127.0.0.1:5000/predict
```

Send patient details as JSON to receive a prediction.

---

## ✅ Conclusion

This project successfully developed a machine learning model for heart disease prediction using a **Random Forest Classifier** and integrated it into a **Flask REST API**. The trained model can process patient information and return predictions in JSON format, demonstrating a simple end-to-end machine learning application. Saving the model with **Joblib** allows it to be reused without retraining, making deployment more efficient. This project also highlights the importance of **MLOps**, where trained models are packaged, version-controlled, and served through APIs for real-world applications. One challenge during deployment is ensuring that the trained model, API, and dependencies remain compatible across different environments.

---

