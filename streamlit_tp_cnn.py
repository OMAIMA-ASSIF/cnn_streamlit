
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# 1. Configuration
st.set_page_config(page_title="SceneVision AI", page_icon="🌍", layout="wide")

SIZE = 150
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# 2. Style
st.markdown("""
<style>
.stApp {background-color: #F5F7FB;}
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {
    background: linear-gradient(120deg,#102A43,#176B87);
    padding: 30px; border-radius: 18px; color: white;
    margin-bottom: 25px;
}
.hero h1 {color: white; font-size: 36px;}
.hero p {color: #D9EAF2;}
[data-testid="stMetric"] {
    background: white; padding: 15px;
    border-radius: 12px; border: 1px solid #E3EAF2;
}
</style>
""", unsafe_allow_html=True)

# 3. Chargement du modèle
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/BestModel2.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# 4. Header
st.markdown("""
<div class="hero">
    <h1>🌍 SceneVision AI</h1>
    <p>Intelligent Image Classification System</p>
    <p>Discover the world through Deep Learning.</p>
</div>
""", unsafe_allow_html=True)

# 5. Upload
st.subheader("📤 Upload your image")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        resized = image.resize((SIZE, SIZE))
        img_array = np.expand_dims(np.array(resized), axis=0)

        # 6. Prediction
        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array, verbose=0)[0]

        if len(predictions) != len(CLASSES):
            st.error("The model must return exactly 6 predictions.")
            st.stop()

        # Assumes the model's final layer uses Softmax
        probabilities = predictions * 100
        index = np.argmax(probabilities)
        predicted_class = CLASSES[index]
        confidence = probabilities[index]

        # 7. Image and result
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width="stretch")

        with col2:
            st.subheader("Prediction Result")
            st.success(f"Predicted class: **{predicted_class.upper()}**")
            st.metric("Confidence Score", f"{confidence:.2f}%")
            st.progress(float(confidence) / 100)

            if confidence >= 80:
                st.success("High confidence")
            elif confidence >= 60:
                st.info("Moderate confidence")
            else:
                st.warning("Low confidence")

        # 8. Probability dataframe
        st.divider()
        st.subheader("Prediction Analytics")

        df = pd.DataFrame({
            "Class": CLASSES,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Class probabilities")
            st.bar_chart(
                df.set_index("Class"),
                horizontal=True,
                x_label="Class",
                y_label="Probability (%)"
            )

        with col4:
            st.markdown("#### Probability distribution")
            st.vega_lite_chart(
                df,
                {
                    "mark": {"type": "arc", "innerRadius": 65},
                    "encoding": {
                        "theta": {"field": "Probability", "type": "quantitative"},
                        "color": {"field": "Class", "type": "nominal"},
                        "tooltip": ["Class", "Probability"]
                    }
                },
                width="stretch"
            )

        # 9. Detailed results
        st.divider()
        st.subheader("🏆 Top 3 Predictions")
        top3 = df.head(3).reset_index(drop=True)

        cols = st.columns(3)
        for i, col in enumerate(cols):
            with col:
                st.metric(
                    f"#{i+1} {top3.loc[i, 'Class'].capitalize()}",
                    f"{top3.loc[i, 'Probability']:.2f}%"
                )

    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    st.info("👆 Upload an image to start classification.")
    st.markdown("**Recognized categories:**")
    st.write("🏢 Buildings | 🌲 Forest | 🧊 Glacier | 🏔️ Mountain | 🌊 Sea | 🛣️ Street")

# 10. Footer
st.divider()
st.caption("SceneVision AI | TensorFlow • Streamlit • Deep Learning")
