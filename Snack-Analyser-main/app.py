import streamlit as st
import numpy as np
import cv2
from PIL import Image

from ocr import adaptive_ocr, smart_product_matcher
from snack_model import load_snack_db, load_health_model, recommend_alternative, detect_category

# ---------------- STREAMLIT DASHBOARD ---------------- #
st.set_page_config(page_title="Snack Health Analyzer", layout="wide")

st.title("🍪 Snack Health Analyzer")
st.markdown("""
Upload an image of a snack product, and get:
- The **recognized product name** via OCR.
- The **predicted health score**.
- A **healthier alternative recommendation** (if available).
""")

# Load DB and model once
snack_db = load_snack_db()
health_model = load_health_model()
known_products = snack_db['Product Name'].tolist()

# ---------------- IMAGE UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload Snack Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("⚠️ Unable to read the image. Please upload a valid PNG/JPG.")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        # ---------------- OCR ---------------- #
        st.subheader("🔍 OCR & Product Recognition")
        ocr_text = adaptive_ocr(uploaded_file)
        st.write(ocr_text if ocr_text else "No text detected.")

        product_name = smart_product_matcher(ocr_text)
        st.write(f"**Detected Product:** {product_name}")

        # ---------------- HEALTH ANALYSIS ---------------- #
        st.subheader("💪 Health Analysis")
        if product_name != "Unknown":
            category = detect_category(product_name)
            st.write(f"Category: {category}")

            # Get predicted health score
            from snack_model import get_health_score
            health_score = get_health_score(product_name, snack_db)
            st.write(f"Predicted Health Score: {health_score}")

            # Recommend healthier alternative
            recommendation = recommend_alternative(product_name, snack_db)
            if recommendation:
                st.write(f"Recommended Product: {recommendation['recommended_product']}")
                st.write(f"Reason: {recommendation['reason']}")
                st.write(f"Predicted Health Gain: {recommendation['predicted_health_gain']}")
            else:
                st.write("No better alternative found.")
        else:
            st.warning("Product not recognized in database.")
