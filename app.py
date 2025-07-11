import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from together import Together

# 🔐 Together.ai API key (set securely in your env, not hardcoded)
TOGETHER_API_KEY = "08e51872d0f3ce01af96b278da8f1f7757e485f71237501ecf42e8cab4661bb8"  # Replace this with your real key

# Initialize Together.ai client
client = Together(api_key= TOGETHER_API_KEY)
# Set device
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

# ───────────── Streamlit UI ─────────────
st.set_page_config(page_title="Caption + Moderation", layout="centered")
st.title("🖼️ Image Captioning + 🛡️ Moderation with Together.ai")

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
        st.warning(f"🛡️ Moderation Result: {result}")

# Manual Text Input
st.subheader("✍️ Or Enter Your Own Text")
text_input = st.text_area("Enter text to check for safety")

if text_input and st.button("🔍 Moderate Text"):
    result = moderate_text_with_together(text_input)
    st.warning(f"🛡️ Moderation Result: {result}")
