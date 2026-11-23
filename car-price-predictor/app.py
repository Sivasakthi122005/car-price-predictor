import base64
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# PATHS (relative to this file, so it runs from anywhere)
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cardata.csv"
MODEL_FILE = BASE_DIR / "models" / "car_price_model_catboost.pkl"
BG_IMAGE = BASE_DIR / "assets" / "background.jpg"


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cardata.csv"
MODEL_FILE = BASE_DIR / "models" / "car_price_model_catboost.pkl"
BG_IMAGE = BASE_DIR / "assets" / "background.jpg"
HOME_IMAGE = BASE_DIR / "assets" / "car.jpg"

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")


# -------------------------------------------------------------
# 1. BACKGROUND IMAGE (embedded as base64 so it works when deployed)
# -------------------------------------------------------------
@st.cache_data
def get_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background_image(image_path: Path):
    if not image_path.exists():
        return
    encoded = get_base64_image(str(image_path))
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# 2. LOGIN CREDENTIALS (from Streamlit secrets, fallback for local)
# -------------------------------------------------------------
def get_credentials():
    try:
        return st.secrets["APP_USERNAME"], st.secrets["APP_PASSWORD"]
    except Exception:
        return "admin", "1234"


# -------------------------------------------------------------
# 3. SESSION STATE
# -------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"


def switch_page(page_name):
    st.session_state.page = page_name


# -------------------------------------------------------------
# 4. LOAD + PREPARE DATA
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=["Unnamed: 0", "Mileage Unit"], errors="ignore")
    df = df.dropna()
    return df


df = load_data().copy()

CAT_COLS = ["name", "fuel", "seller_type", "transmission", "owner"]
for col in CAT_COLS:
    df[col] = df[col].astype("category")

name_reverse_mapping = {v: k for k, v in enumerate(df["name"].cat.categories)}
fuel_reverse_mapping = {v: k for k, v in enumerate(df["fuel"].cat.categories)}
seller_reverse_mapping = {v: k for k, v in enumerate(df["seller_type"].cat.categories)}
trans_reverse_mapping = {v: k for k, v in enumerate(df["transmission"].cat.categories)}
owner_reverse_mapping = {v: k for k, v in enumerate(df["owner"].cat.categories)}

for col in CAT_COLS:
    df[col] = df[col].cat.codes

X = df[["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
y = df["selling_price"]


# -------------------------------------------------------------
# 5. LOAD (or TRAIN) MODEL
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass  # incompatible/corrupt pickle -> retrain below

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42,
        allow_writing_files=False,
    )
    mdl.fit(X_train, y_train, cat_features=[0, 3, 4, 5, 6])
    try:
        MODEL_FILE.parent.mkdir(exist_ok=True)
        joblib.dump(mdl, MODEL_FILE)
    except Exception:
        pass  # read-only filesystem on some hosts; keep the in-memory model
    return mdl


model = load_model()

set_background_image(BG_IMAGE)


# -------------------------------------------------------------
# 6. LOGIN PAGE
# -------------------------------------------------------------
if st.session_state.page == "login":
    st.title("🔐 Login to Car Price Predictor")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        valid_user, valid_pass = get_credentials()
        if username == valid_user and password == valid_pass:
            switch_page("home")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# -------------------------------------------------------------
# 7. HOME PAGE
# -------------------------------------------------------------
elif st.session_state.page == "home":
    set_background_image(HOME_IMAGE)
    st.title("🚗 Welcome to Car Price Predictor!")
    st.markdown("### Choose an option below 👇")

    if st.button("Start Prediction"):
        switch_page("prediction")
        st.rerun()

    if st.button("Logout"):
        switch_page("login")
        st.rerun()

# -------------------------------------------------------------
# 8. PREDICTION PAGE
# -------------------------------------------------------------
elif st.session_state.page == "prediction":
    st.title("🚘 Car Price Prediction")
    st.sidebar.header("Enter Car Details")

    car_name = st.sidebar.selectbox("Car Name", sorted(name_reverse_mapping.keys()))
    year = st.sidebar.number_input(
        "Car Year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        step=1,
    )
    kms_driven = st.sidebar.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=int(df["km_driven"].max()),
        step=500,
    )
    fuel = st.sidebar.selectbox("Fuel Type", sorted(fuel_reverse_mapping.keys()))
    seller_type = st.sidebar.selectbox("Seller Type", sorted(seller_reverse_mapping.keys()))
    transmission = st.sidebar.selectbox("Transmission", sorted(trans_reverse_mapping.keys()))
    owner = st.sidebar.selectbox("Previous Owner Type", sorted(owner_reverse_mapping.keys()))

    if st.sidebar.button("Predict Price"):
        if year < 1990 or year > 2025:
            st.error("⚠️ Enter a valid year between 1990 and 2025.")
        else:
            input_data = pd.DataFrame(
                [[
                    name_reverse_mapping[car_name],
                    year,
                    kms_driven,
                    fuel_reverse_mapping[fuel],
                    seller_reverse_mapping[seller_type],
                    trans_reverse_mapping[transmission],
                    owner_reverse_mapping[owner],
                ]],
                columns=X.columns,
            )

            base_price = model.predict(input_data)[0]

            # Depreciation based on previous owner
            owner_factor_dict = {
                "First Owner": 1.0,
                "Second Owner": 0.7,
                "Third Owner": 0.5,
                "Fourth & Above": 0.4,
            }
            adjusted_price = base_price * owner_factor_dict.get(owner, 1.0)

            # Simulated +/-5% uncertainty over 30 days
            price_std = adjusted_price * 0.05
            daily_fluctuations = np.random.normal(0, price_std / adjusted_price, 30)
            future_prices = adjusted_price * (1 + daily_fluctuations)
            best_day = int(np.argmin(future_prices)) + 1
            best_price = future_prices[best_day - 1]

            def format_price(amount):
                if amount >= 1e7:
                    return round(amount / 1e7, 2), "Cr"
                return round(amount / 1e5, 2), "Lakh"

            base_val, base_unit = format_price(adjusted_price)
            std_low, _ = format_price(adjusted_price - price_std)
            std_high, _ = format_price(adjusted_price + price_std)
            future_vals = [format_price(p)[0] for p in future_prices]
            best_val, best_unit = format_price(best_price)

            st.sidebar.success(f"💰 Estimated Price: ₹ {base_val} {base_unit}")
            st.sidebar.info(f"📉 Price Range (±5%): ₹ {std_low} - ₹ {std_high} {base_unit}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 Next 30-Day Price Predictions")
                st.table(
                    pd.DataFrame(
                        {"Day": range(1, 31), f"Predicted Price (₹ {base_unit})": future_vals}
                    )
                )

            with col2:
                st.subheader("📈 Price Trend")
                fig, ax = plt.subplots(figsize=(6, 4))
                days = range(1, 31)
                ax.plot(days, future_vals, marker="o", linestyle="-", color="b", alpha=0.7)
                ax.fill_between(
                    days,
                    [v * 0.95 for v in future_vals],
                    [v * 1.05 for v in future_vals],
                    color="b",
                    alpha=0.2,
                    label="±5%",
                )
                ax.axvline(best_day, color="r", linestyle="--", label=f"Best Day: {best_day}")
                ax.scatter(best_day, best_val, color="red",
                           label=f"₹ {best_val} {best_unit}", zorder=3)
                ax.set_xlabel("Days")
                ax.set_ylabel(f"Price (₹ {base_unit})")
                ax.set_title("Car Price Trend Over 30 Days")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            st.success(f"✅ Best day to buy: **Day {best_day}** at ₹ {best_val} {best_unit}")

    if st.button("⬅️ Back to Home"):
        switch_page("home")
        st.rerun()
