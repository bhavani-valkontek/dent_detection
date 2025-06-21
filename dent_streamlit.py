# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import json
from PIL import Image

# ===============================
# 🎯 Function: Run YOLO Inference
# ===============================
def run_inference(image_path, model_path, conf_threshold):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf_threshold, imgsz=640, save=False)
    output = results[0]

    image = cv2.imread(image_path)
    image_draw = image.copy()
    detection_data = []

    if output.boxes is not None and len(output.boxes) > 0:
        for box in output.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            class_name = output.names[int(cls)]

            # Draw bounding box
            cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image_draw, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detection_data.append({
                "class": class_name,
                "confidence": float(f"{conf:.4f}"),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })
    return image_draw, detection_data

# ============================
# 🚀 Streamlit Web Application
# ============================
st.set_page_config(page_title="YOLOv8 Dent Detection", layout="centered")
st.title("🔍 Dent Detection using YOLOv8")
st.markdown("Upload an image and let the model detect car dents.")

# Upload model and image
model_file = "best _model.pt"
image_file = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("🎯 Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

# Run detection
if model_file and image_file:
    with st.spinner("⏳ Running Detection..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(model_file.read())
            tmp_model_path = tmp_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_file.read())
            tmp_img_path = tmp_img.name

        output_image, detections = run_inference(tmp_img_path, tmp_model_path, conf_threshold)

        # Show detections
        st.image(output_image, caption="🖼️ Detected Image", channels="BGR", use_column_width=True)

        st.subheader("📊 Detection Results")
        if detections:
            st.json(detections)

            # Save detection results as JSON
            json_path = "detection_results.json"
            with open(json_path, "w") as f:
                json.dump(detections, f, indent=2)
            st.download_button("⬇️ Download Results (JSON)", data=json.dumps(detections, indent=2),
                               file_name="detection_results.json", mime="application/json")
        else:
            st.warning("❌ No dents detected in the image.")

        # Save output image
        output_image_path = "dent_detection_output.jpg"
        cv2.imwrite(output_image_path, output_image)

        # Offer download of output image
        with open(output_image_path, "rb") as img_file:
            st.download_button("⬇️ Download Output Image", img_file.read(),
                               file_name="dent_detection_output.jpg", mime="image/jpeg")
