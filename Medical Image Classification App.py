<<<<<<< HEAD
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page setup
st.set_page_config(page_title="Disease Detection", layout="wide")
st.title("🩺 Medical Image Classification App")

# Sidebar - Choose the model
model_choice = st.sidebar.selectbox("🔍 Choose a Model", ["Brain Tumor", "Chest X-Ray (Pneumonia)"])

# Load the selected model
if model_choice == "Brain Tumor":
    model = load_model("brain_tumor.h5")
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    img_size = (224, 224)
    st.sidebar.markdown("""
    ### Information:
    - **Glioma**: Brain tissue tumor
    - **Meningioma**: Brain membrane tumor
    - **Pituitary**: Hormonal center tumor
    - **No Tumor**: Healthy brain
    """)
elif model_choice == "Chest X-Ray (Pneumonia)":
    model = load_model("chest-xray-pneumonia.h5")
    class_names = ['NORMAL', 'PNEUMONIA']
    img_size = (224, 224)
    st.sidebar.markdown("""
    ### Information:
    - **NORMAL**: No signs of infection
    - **PNEUMONIA**: Signs of lung infection
    """)

# Image preprocessing function
def preprocess_uploaded(img_file, img_size):
    img = Image.open(img_file).convert('RGB')
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("🔍 Predict"):
        img = preprocess_uploaded(uploaded_file, img_size)
        prediction = model.predict(img)

        if model_choice == "Brain Tumor":
            pred_index = np.argmax(prediction)
            pred_class = class_names[pred_index]
            confidence = round(float(np.max(prediction)) * 100, 2)
            if pred_class == "notumor":
                st.success(f"✅ Result: No Tumor ({confidence}%)")
            else:
                st.error(f"⚠️ Tumor Detected: {pred_class.upper()} ({confidence}%)")

        elif model_choice == "Chest X-Ray (Pneumonia)":
            pred_class = class_names[int(prediction[0][0] > 0.5)]
            confidence = round(float(prediction[0][0] if pred_class == "PNEUMONIA" else 1 - prediction[0][0]) * 100, 2)
            if pred_class == "NORMAL":
                st.success(f"✅ Result: Normal ({confidence}%)")
            else:
                st.error(f"⚠️ Pneumonia Detected ({confidence}%)")
=======
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page setup
st.set_page_config(page_title="Disease Detection", layout="wide")
st.title("🩺 Medical Image Classification App")

# Sidebar - Choose the model
model_choice = st.sidebar.selectbox("🔍 Choose a Model", ["Brain Tumor", "Chest X-Ray (Pneumonia)"])

# Load the selected model
if model_choice == "Brain Tumor":
    model = load_model("D:/DLProjects/Diseases/brain_tumor.h5")
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    img_size = (224, 224)
    st.sidebar.markdown("""
    ### Information:
    - **Glioma**: Brain tissue tumor
    - **Meningioma**: Brain membrane tumor
    - **Pituitary**: Hormonal center tumor
    - **No Tumor**: Healthy brain
    """)
elif model_choice == "Chest X-Ray (Pneumonia)":
    model = load_model("D:/DLProjects/Diseases/chest-xray-pneumonia.h5")
    class_names = ['NORMAL', 'PNEUMONIA']
    img_size = (224, 224)
    st.sidebar.markdown("""
    ### Information:
    - **NORMAL**: No signs of infection
    - **PNEUMONIA**: Signs of lung infection
    """)

# Image preprocessing function
def preprocess_uploaded(img_file, img_size):
    img = Image.open(img_file).convert('RGB')
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("🔍 Predict"):
        img = preprocess_uploaded(uploaded_file, img_size)
        prediction = model.predict(img)

        if model_choice == "Brain Tumor":
            pred_index = np.argmax(prediction)
            pred_class = class_names[pred_index]
            confidence = round(float(np.max(prediction)) * 100, 2)
            if pred_class == "notumor":
                st.success(f"✅ Result: No Tumor ({confidence}%)")
            else:
                st.error(f"⚠️ Tumor Detected: {pred_class.upper()} ({confidence}%)")

        elif model_choice == "Chest X-Ray (Pneumonia)":
            pred_class = class_names[int(prediction[0][0] > 0.5)]
            confidence = round(float(prediction[0][0] if pred_class == "PNEUMONIA" else 1 - prediction[0][0]) * 100, 2)
            if pred_class == "NORMAL":
                st.success(f"✅ Result: Normal ({confidence}%)")
            else:
                st.error(f"⚠️ Pneumonia Detected ({confidence}%)")
>>>>>>> 37bbbc6 (first commit)
