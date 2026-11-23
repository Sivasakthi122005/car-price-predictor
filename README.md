# 🚗 Car Price Predictor

A machine learning web app that estimates the resale price of a used car in India.

🔗 **Live demo:** [Open the app](https://sivasakthi122005-car-price-predic-car-price-predictorapp-kjeo29.streamlit.app/)

## What is this project?
Buying or selling a used car is hard because it is difficult to know a fair price. This app solves that: enter a car's details and it instantly estimates the price in ₹ Lakh or Crore.

## Features
- Login page and home screen with a custom background
- Prediction from 7 inputs: car name, year, kilometers driven, fuel type, seller type, transmission and previous owner type
- Estimated price with a ±5% price range
- Owner-based depreciation adjustment
- 30-day simulated price table, trend chart and "best day to buy" highlight

## How it works
1. **Data:** `cardata.csv` holds used-car listings. Unused columns are dropped and rows with missing values are removed.
2. **Encoding:** Text columns (name, fuel, seller type, transmission, owner) are converted to categorical codes.
3. **Model:** A CatBoost regressor (1000 iterations, learning rate 0.1, depth 6) predicts the selling price. The trained model is saved as a `.pkl` file and loaded at startup. If it cannot be loaded, the app retrains it automatically.
4. **Prediction:** Inputs are encoded the same way, passed to the model, and the output is formatted in Lakh or Crore.
5. **Visualization:** Matplotlib draws the 30-day trend with a shaded ±5% band.

## Tech stack
Python, Streamlit, Pandas, NumPy, CatBoost, scikit-learn, Joblib, Matplotlib, Docker

## Project structure
```
├── app.py
├── requirements.txt
├── Dockerfile
├── data/cardata.csv
├── models/car_price_model_catboost.pkl
└── assets/   (background images)
```

## Run locally
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Default local login: `admin` / `1234`. For deployment, set `APP_USERNAME` and `APP_PASSWORD` in Streamlit secrets.

## Limitations
- The 30-day price trend is a simulation (random noise around the predicted price), not a real time-series forecast.
- Predictions depend on the quality and coverage of the dataset.

## Future improvements
- Show model accuracy (R², MAE) and a feature-importance chart
- Replace the simulated trend with real historical price data
- Add proper user authentication
