import os, shutil, zipfile, textwrap

project_dir = "/mnt/data/house_price_prediction_app"
os.makedirs(project_dir, exist_ok=True)

app_code = r'''
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------- Page configuration ---------------------------
st.set_page_config(
    page_title="HomeValue | Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "linear_regression.pkl"

# ------------------------------- Styling ---------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 90% 3%, rgba(72, 187, 170, .12), transparent 26%),
            linear-gradient(135deg, #f7fafc 0%, #eef4f8 100%);
        color: #14213d;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101d35 0%, #172b4d 100%);
    }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    [data-testid="stSidebar"] .stNumberInput input {
        background: #fff;
        color: #14213d !important;
    }
    .hero {
        padding: 30px 32px;
        border-radius: 24px;
        background: linear-gradient(120deg, #13233f 0%, #21476a 62%, #167d80 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(18, 39, 69, .16);
        margin-bottom: 24px;
    }
    .hero-kicker {
        color: #a9f0df;
        text-transform: uppercase;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .16em;
        margin-bottom: 10px;
    }
    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.1;
        margin: 0 0 12px 0;
    }
    .hero p {
        color: #e1edf5;
        font-size: 1.02rem;
        max-width: 760px;
        margin: 0;
    }
    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        color: #172b4d;
        margin: 8px 0 4px 0;
    }
    .section-subtitle { color: #61728a; margin-bottom: 18px; }
    .result-card {
        padding: 25px;
        border-radius: 22px;
        background: white;
        border: 1px solid #dce7ee;
        box-shadow: 0 12px 35px rgba(31, 55, 82, .08);
    }
    .result-label {
        color: #65758b;
        font-size: .84rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .09em;
    }
    .result-price {
        color: #087f72;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.15;
        margin: 10px 0;
        overflow-wrap: anywhere;
    }
    .result-note { color: #64748b; font-size: .9rem; }
    div.stButton > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        padding: .78rem 1rem;
        background: linear-gradient(90deg, #087f72, #149b8c);
        color: white;
        font-weight: 700;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(8, 127, 114, .22);
    }
    .info-card {
        background: rgba(255,255,255,.82);
        border: 1px solid #dce7ee;
        border-radius: 16px;
        padding: 17px;
        height: 100%;
    }
    .info-card strong { color: #14213d; }
    .footer {
        color: #718096;
        font-size: .82rem;
        text-align: center;
        padding: 26px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Load model ------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

model = load_model()

# These names and order match the uploaded LinearRegression model.
FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

# -------------------------------- Header ----------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Real estate • Machine learning</div>
        <h1>Find your home's estimated value.</h1>
        <p>Explore a data-driven estimate using property size, rooms, build year,
        lot size, garage capacity, and neighborhood quality.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model file not found. Place `linear_regression.pkl` in the same folder as `app.py`."
    )
    st.stop()

if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
    st.error("The model does not expect 7 features. Please check the model file and feature list.")
    st.stop()

# -------------------------------- Inputs ----------------------------------
st.markdown('<div class="section-heading">Property details</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Enter the details below, then select “Estimate home value”.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 1], gap="large")

with st.sidebar:
    st.markdown("## 🏡 HomeValue")
    st.caption("Property profile")
    st.markdown("---")
    square_footage = st.number_input(
        "Living area (square feet)", min_value=100, max_value=50000,
        value=1800, step=100, help="Total home area in square feet."
    )
    bedrooms = st.number_input(
        "Bedrooms", min_value=0, max_value=20, value=3, step=1
    )
    bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5
    )
    year_built = st.number_input(
        "Year built", min_value=1800, max_value=2100, value=2010, step=1
    )
    lot_size = st.number_input(
        "Lot size", min_value=0.0, max_value=100000.0, value=0.25, step=0.05,
        help="Use the same unit (and scale) used in the dataset when training the model."
    )
    garage_size = st.number_input(
        "Garage size", min_value=0, max_value=20, value=2, step=1,
        help="Enter this using the same unit used in the training dataset."
    )
    neighborhood_quality = st.slider(
        "Neighborhood quality score", min_value=1, max_value=10, value=7,
        help="Use the same scoring scale as the model's training data."
    )
    st.markdown("---")
    estimate_clicked = st.button("✨ Estimate home value", use_container_width=True)
    st.caption("Prediction is an estimate, not a formal property valuation.")

with left:
    st.markdown("### Your property at a glance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Living area", f"{square_footage:,.0f} sq ft")
    m2.metric("Bedrooms", f"{bedrooms:.0f}")
    m3.metric("Bathrooms", f"{bathrooms:g}")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-card">
            <strong>📍 Property profile</strong><br><br>
            <span style="color:#61728a">Built in</span> <strong>{int(year_built)}</strong><br>
            <span style="color:#61728a">Lot size</span> <strong>{lot_size:g}</strong><br>
            <span style="color:#61728a">Garage size</span> <strong>{int(garage_size)}</strong><br>
            <span style="color:#61728a">Neighborhood score</span> <strong>{neighborhood_quality}/10</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Important: Lot size, garage size, and neighborhood quality must use the same "
        "units/scales as the data used to train your model. Adjust the default values if your dataset uses different units."
    )

with right:
    st.markdown("### Estimated value")
    if estimate_clicked:
        input_df = pd.DataFrame(
            [[
                square_footage,
                bedrooms,
                bathrooms,
                year_built,
                lot_size,
                garage_size,
                neighborhood_quality,
            ]],
            columns=FEATURES,
        )
        try:
            prediction = float(model.predict(input_df)[0])
            if not np.isfinite(prediction):
                st.error("The model returned an invalid value. Please check the input values.")
            else:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted home value</div>
                        <div class="result-price">${prediction:,.0f}</div>
                        <div class="result-note">Model estimate based on the property details you entered.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Estimated value per sq ft", f"${prediction / square_footage:,.2f}")
                st.caption("Currency is displayed as USD by default. Change the display if your training data uses another currency.")
                with st.expander("View the values sent to the model"):
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")
    else:
        st.markdown(
            """
            <div class="result-card">
                <div style="font-size:2.2rem">🏠</div>
                <div class="result-label">Ready when you are</div>
                <div style="font-family:Manrope;font-size:1.35rem;font-weight:800;color:#172b4d;margin:10px 0">
                    Your estimate will appear here
                </div>
                <div class="result-note">Fill in the property details in the sidebar and click the estimate button.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <strong>How it works</strong><br><br>
                1. Enter property features.<br>
                2. The saved Linear Regression model processes them.<br>
                3. The app displays the estimated home value.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
a, b, c = st.columns(3)
with a:
    st.markdown('<div class="info-card">📊<br><strong>ML-powered</strong><br><span style="color:#61728a">Linear Regression model</span></div>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="info-card">⚡<br><strong>Instant estimate</strong><br><span style="color:#61728a">Predict directly in the app</span></div>', unsafe_allow_html=True)
with c:
    st.markdown('<div class="info-card">🔎<br><strong>Transparent inputs</strong><br><span style="color:#61728a">Review features sent to the model</span></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">HomeValue • Educational machine-learning project • Not financial or appraisal advice</div>',
    unsafe_allow_html=True,
)
'''
requirements = """streamlit>=1.40,<2.0
pandas>=2.0,<3.0
numpy>=1.26,<3.0
scikit-learn==1.9.0
"""
readme = """# HomeValue — House Price Prediction

A Streamlit web app using a saved scikit-learn Linear Regression model to estimate home values from seven property features.

## Files

- `app.py` — Streamlit application and custom UI
- `linear_regression.pkl` — trained model
- `requirements.txt` — Python dependencies

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Model inputs

The app sends features to the model in this exact order:

1. `Square_Footage`
2. `Num_Bedrooms`
3. `Num_Bathrooms`
4. `Year_Built`
5. `Lot_Size`
6. `Garage_Size`
7. `Neighborhood_Quality`

Make sure lot size, garage size, and neighborhood quality use the same units and scales as the data used to train the model. The UI displays the prediction in USD; change that display if the training target uses a different currency.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Deploy.

## Deploy on Render (alternative)

Create a **Web Service** connected to the GitHub repository.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Note

This is an educational estimate, not a professional property appraisal. Only load pickle files you trust, because pickle files can execute code when deserialized.
"""
with open(os.path.join(project_dir, "app.py"), "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(app_code).strip() + "\n")
with open(os.path.join(project_dir, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(requirements)
with open(os.path.join(project_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme)
shutil.copy2("/mnt/data/linear_regression.pkl", os.path.join(project_dir, "linear_regression.pkl"))

zip_path = "/mnt/data/house_price_prediction_github_ready.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename in ["app.py", "requirements.txt", "README.md", "linear_regression.pkl"]:
        zf.write(os.path.join(project_dir, filename), arcname=f"house_price_prediction_app/{filename}")

print("Created project files:")
print(os.path.join(project_dir, "app.py"))
print(os.path.join(project_dir, "requirements.txt"))
print(os.path.join(project_dir, "README.md"))
print(zip_path)
