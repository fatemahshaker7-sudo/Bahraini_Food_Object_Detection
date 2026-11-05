import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io

st.set_page_config(
    page_title="Bahraini Food Detection",
    page_icon="🍔",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

st.title("🪙Bahraini Food Detection🪙")

st.write(
    "Detect Bahraini Food using a YOLO26 deep learning model."
)

model_path = Path("best.pt")

if not model_path.exists():
    st.error("Model file 'best.pt' not found.")
    st.stop()

model = load_model()

# Choose input method
option = st.radio(
    "Choose an option:",
    ["Upload Image", "Take a Photo"]
)

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a currency image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input(
        "Take a photo of Bahraini currency"
    )

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

    if st.button("Detect Currency"):

        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=224,
            verbose=False
        )

        result = results[0]

        st.subheader("Detection Result")

        st.image(
            result.plot(),
            channels="BGR",
            use_container_width=True
        )

        if len(result.boxes) > 0:

            st.success("Food detected!")

            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                currency = model.names[class_id]

                st.write(
                    f"**Food:** {currency}"
                )
                st.write(
                    f"**Confidence:** {score:.2%}"
                )

            # Download detection image
            annotated = Image.fromarray(
                result.plot()[..., ::-1]
            )

            buffer = io.BytesIO()
            annotated.save(buffer, format="PNG")

            st.download_button(
                "Download Result",
                data=buffer.getvalue(),
                file_name="currency_detection.png",
                mime="image/png"
            )

        else:
            st.warning("No Food detected.")
