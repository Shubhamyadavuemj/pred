import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Load the model and scaler globally
url = 'https://drive.google.com/uc?id=1rrNikVGWYxUiDvryTygEJAeU0Z5QF2np'
df = pd.read_csv(url)
predictors = df.drop("target", axis=1)
target = df["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
conf_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


@app.route('/api/predict', methods=['GET'])
def predict():
    data = request.get_json()
    p1 = request.args.get('p1', default=None, type=str)
    p2 = request.args.get('p2', default=None, type=str)
    p3 = request.args.get('p3', default=None, type=str)
    p4 = request.args.get('p4', default=None, type=str)
    p5 = request.args.get('p5', default=None, type=str)
    p6 = request.args.get('p6', default=None, type=str)
    p7 = request.args.get('p7', default=None, type=str)
    p8 = request.args.get('p8', default=None, type=str)
    p9 = request.args.get('p9', default=None, type=str)
    p10 = request.args.get('p10', default=None, type=str)
    p11 = request.args.get('p11', default=None, type=str)
    p12 = request.args.get('p12', default=None, type=str)
    p13 = request.args.get('p13', default=None, type=str)
   
    
    new_data = [[int(p1),int(p2),int(p3),int(p4),int(p5),int(p6),int(p7),int(p8),int(p9),int(p10),int(p11),int(p12),int(p13)]] 
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    
    
    
    number_str = str(prediction[0])
    number_json = json.dumps(number_str)
    
    return number_json
    

@app.route('/api/health', methods=['GET'])
def health_check():
    # Simple health check endpoint to ensure the API is up
    return jsonify({'status': 'API is running'}), 200

if __name__ == '__main__':
    app.run()
    
    
#http://127.0.0.1:5000/api/predict/p1=58&p2=0&p3=0&p4=100&p5=248&p6=0&p7=0&p8=122&p9=0&p10=1&p11=1&p12=0&p13=2
