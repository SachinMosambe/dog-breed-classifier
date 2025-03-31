import os
import json
import streamlit as st
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image

# Fixed Paths
MODEL_PATH = "D:/AI engineer project/Dog_breed_classifier/model/dog_breed_classifier.pth"
CLASS_LABELS_PATH = "D:/AI engineer project/Dog_breed_classifier/model/class_labels.json"

# Ensure `class_labels.json` exists
if not os.path.exists(CLASS_LABELS_PATH):
    st.error(f"Error: `class_labels.json` not found at {CLASS_LABELS_PATH}")
    st.stop()

# Load class labels
with open(CLASS_LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# Function to clean breed names (Remove initial ID)
def clean_breed_name(raw_name):
    return raw_name.split("-")[-1].replace("_", " ")  # Removes ID & replaces underscores with spaces

# Use `st.cache_resource` instead of `st.cache`
@st.cache_resource
def load_model():
    model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=len(class_labels))
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        st.stop()

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit UI
st.title("Dog Breed Classifier")
st.markdown("Upload an image of a dog to classify its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Fix UI Warning: Use `use_container_width`
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = transform(image).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 2 predictions
        top2_indices = torch.topk(probabilities, 2).indices.numpy()
        top2_confidences = torch.topk(probabilities, 2).values.numpy()

        # Map predictions to clean breed names
        top2_breeds = [clean_breed_name(class_labels[str(idx)]) for idx in top2_indices]
        top2_confidence_scores = [conf * 100 for conf in top2_confidences]

    # Display Top 2 Predictions
    st.success(f"Top Prediction: {top2_breeds[0]} ({top2_confidence_scores[0]:.2f}%)")
    st.info(f"Second Prediction: {top2_breeds[1]} ({top2_confidence_scores[1]:.2f}%)")

