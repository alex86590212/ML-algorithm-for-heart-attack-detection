import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('C:/Users/alexa/OneDrive/Desktop/uni improg/data.csv')


X = data[['Age','Coronary Artery Disease', 'Hypertension', 'Diabetes','Hyperlipidemia','Obesity','Smoking','Family History of Heart Disease','Chronic Kidney Disease ','Chronic Obstructive Pulmonary Disease','Rheumatoid Arthritis']]
y = data['HeartAttack']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)


new_example = pd.DataFrame({
    'Age': [78], 
    'Coronary Artery Disease': [0],
    'Hypertension': [1],
    'Diabetes': [0],
    'Hyperlipidemia': [0],
    'Obesity': [0],
    'Smoking': [0],
    'Family History of Heart Disease': [1],
    'Chronic Kidney Disease ':[0],
    'Chronic Obstructive Pulmonary Disease':[0],
    'Rheumatoid Arthritis': [0]
      
    
})

new_example_scaled = scaler.transform(new_example)

heart_attack_probability = model.predict_proba(new_example_scaled)[:, 1]


print(f'Probability of a Heart Attack: {heart_attack_probability[0]:.2f}')
