"""
Streamlit demo. Run with: streamlit run app/app.py
Upload a chest X-ray -> see the highlighted region, the generated report,
and the triage/urgency recommendation.
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import run_pipeline  # noqa: E402

st.set_page_config(page_title="Chest X-ray Diagnostic Assistant", layout="wide")
st.title("Explainable Chest X-ray Diagnostic Assistant")
st.caption(
    "Prototype for offline, low-resource clinics. This is a decision-support "
    "second opinion, not a replacement for a qualified clinician."
)

uploaded = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

checkpoint_path = "models/chest_classifier.pt"

if uploaded is not None:
    if not os.path.exists(checkpoint_path):
        st.error(
            f"No trained model found at {checkpoint_path}. Run `python src/train.py` "
            f"first (see README)."
        )
    else:
        temp_path = "outputs/_uploaded_temp.png"
        os.makedirs("outputs", exist_ok=True)
        Image.open(uploaded).convert("L").save(temp_path)

        with st.spinner("Analyzing..."):
            result = run_pipeline(temp_path, checkpoint=checkpoint_path)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(Image.open(temp_path), use_container_width=True)
        with col2:
            st.subheader("Highlighted region of concern")
            st.image(result["overlay_image"], use_container_width=True)

        tier = result["triage"]["tier"]
        tier_color = {"Routine": "green", "Soon": "blue", "Urgent": "orange", "Immediate": "red"}
        st.markdown(
            f"### Triage: :{tier_color.get(tier, 'gray')}[{tier}]"
        )
        st.write(result["triage"]["message"])

        st.subheader("Generated report")
        st.write(result["report"])

        st.caption(
            f"Predicted: {result['condition']} | "
            f"Confidence: {result['confidence']*100:.1f}% | "
            f"Region: {result['region']}"
        )
else:
    st.info("Upload an X-ray image to get started.")