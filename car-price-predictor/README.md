# 🚗 Car Price Predictor

Streamlit app that predicts used-car prices with a CatBoost model.

## Project structure
```
car-price-predictor/
├── app.py                        # Streamlit app (UI + prediction logic)
├── requirements.txt              # Pinned dependencies
├── Dockerfile                    # For Render / Railway / any Docker host
├── data/cardata.csv              # Training / lookup data
├── models/car_price_model_catboost.pkl
├── assets/background.jpg
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example      # Copy -> secrets.toml for your login
```

## Run locally
```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Default login (local only): `admin` / `1234`.
To change it, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and edit.

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo (do NOT commit `secrets.toml`).
2. Go to share.streamlit.io -> New app -> select repo, branch, main file `app.py`.
3. Advanced settings -> Python 3.12 (or 3.13) and paste into **Secrets**:
   ```
   APP_USERNAME = "your-username"
   APP_PASSWORD = "your-strong-password"
   ```
4. Deploy.

## Deploy with Docker (Render / Railway / etc.)
```
docker build -t car-price-predictor .
docker run -p 8501:8501 car-price-predictor
```
On Render/Railway pick "Docker"; the app reads the `PORT` env variable automatically.

## Notes
- The model was trained with catboost 1.2.8 (pinned). If loading ever fails, the app retrains automatically.
- The 30-day "future prices" are simulated random noise around the predicted price, not a real forecast.
