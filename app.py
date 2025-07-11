import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from together import Together
import os

from huggingface_hub import hf_hub_download
from transformers import BertTokenizer
import onnxruntime as ort
import numpy as np
import joblib

# Constants
REPO_ID = "M-Montasser/finetuned-distilbert"
MODEL_FILENAME = "model.onnx"
ENCODER_FILENAME = "label_encoder.pkl"

# Download files directly from Hugging Face Hub
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
encoder_path = hf_hub_download(repo_id=REPO_ID, filename=ENCODER_FILENAME)
tokenizer = BertTokenizer.from_pretrained(REPO_ID)

# Load ONNX model
session = ort.InferenceSession(model_path)

# Load Label Encoder
label_encoder = joblib.load(encoder_path)


os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
# Initialize Together.ai client
client = Together()# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Load BLIP model (for captioning)
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    ).to(device)
    return processor, model

# Generate image caption
def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Use Together.ai to moderate the text
def moderate_text_with_together(text):
    response = client.chat.completions.create(
        model="meta-llama/LlamaGuard-2-8b",
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content


def verify_with_distilbert(text):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )

    ort_inputs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }

    logits = session.run(None, ort_inputs)[0]
    predicted_class = np.argmax(logits, axis=1)
    label = label_encoder.inverse_transform(predicted_class)[0]
    return label


#  Streamlit UI 
st.set_page_config(page_title="Caption + Moderation", layout="centered")
st.title("🖼️ Image Captioning + 🛡️ Moderation")

processor, caption_model = load_blip()

# Image Upload
st.subheader("📷 Upload an Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("📝 Generate Caption and Moderate"):
        caption = generate_caption(image, processor, caption_model)
        st.success(f"📋 Caption: {caption}")
        result = moderate_text_with_together(caption)
        if "unsafe" in result.lower():
            st.warning(f"🛡️ Moderation Result: {result}")
            st.error("🚫 Text marked UNSAFE by LLaMA Guard")
        else:
            st.warning(f"🛡️ Moderation Result: {result}")
            st.success("✅ Text marked safe by LLaMA Guard")
            label = verify_with_distilbert(caption)
            st.info(f"🔁 DistilBERT verification: {label}")

# Manual Text Input
st.subheader("✍️ Or Enter Your Own Text")
text_input = st.text_area("Enter text to check for safety")

if text_input and st.button("🔍 Moderate Text"):
    result = moderate_text_with_together(text_input)
    if "unsafe" in result.lower():
        st.warning(f"🛡️ Moderation Result: {result}")
        st.error("🚫 Text marked UNSAFE by LLaMA Guard")

    else:
        st.warning(f"🛡️ Moderation Result: {result}")
        st.success("✅ Text marked safe by LLaMA Guard")
        label = verify_with_distilbert(text_input)
        st.info(f"🔁 DistilBERT verification: {label}")

