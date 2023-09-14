import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('fruits_vegetables_prices2.csv')
data.head()
data.tail()
features = ['Commodity','Minimum','Maximum','Average']
target = 'Average'
label_encoder = LabelEncoder()
data['Commodity'] = label_encoder.fit_transform(data['Commodity'])
X = data[features]
y = data[target]
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
def predict_price(commodity_name ,minimumprice, maximumprice, averageprice):
    commodity_encoded = label_encoder.transform([commodity_name])[0]
    scaled_features = scaler.transform([[commodity_encoded, minimumprice, maximumprice, averageprice]])
    prediction = model.predict(scaled_features)
    return prediction[0]

import joblib

joblib.dump(model, 'prediction_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

