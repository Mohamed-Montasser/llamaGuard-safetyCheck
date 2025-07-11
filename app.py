import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
from huggingface_hub import login, HfFolder
# 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
# 🔐 Hugging Face Token (Replace with your own valid token)
HF_TOKEN = "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"

# Authenticate with Hugging Face
try:
    login(token=HF_TOKEN, add_to_git_credential=False)
    HfFolder.save_token(HF_TOKEN)
except Exception as e:
    st.error(f"Authentication failed: {str(e)}")
    st.stop()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Load BLIP model for image captioning
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=dtype
    ).to(device)
    return processor, model

# Load LLaMA Guard model and tokenizer
@st.cache_resource
def load_llama_guard():
    model_id = "meta-llama/LlamaGuard-7b"

    # Load config and patch rope_scaling
    config = AutoConfig.from_pretrained(
        model_id,
        token=HF_TOKEN,
        trust_remote_code=True
    )

    # Patch rope_scaling to avoid validation crash
    if isinstance(config.rope_scaling, dict):
        config.rope_scaling = {
            "type": "linear",  # could be "dynamic" if specified by model card
            "factor": config.rope_scaling.get("factor", 8.0)
        }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN, device_map="auto", torch_dtype=torch.float16)


    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# Captioning function
def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Moderation function
def moderate_text(text, tokenizer, model):
    chat = [{"role": "user", "content": text}]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = (input_ids != pad_token_id).long().to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = input_ids.shape[-1]
    result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return result

# Streamlit UI
st.set_page_config(page_title="Caption + Moderation", layout="centered")
st.title("🖼️ Image Captioning + 🛡️ Safety Check with LLaMA Guard")

# Load models
processor, caption_model = load_blip()
tokenizer, llama_guard_model = load_llama_guard()

# Image upload section
st.subheader("📷 Upload an Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("📝 Generate Caption and Moderate"):
        caption = generate_caption(image, processor, caption_model)
        st.success(f"📋 Caption: {caption}")

        result = moderate_text(caption, tokenizer, llama_guard_model)
        st.warning(f"🛡️ Moderation Result: {result}")

# Text input moderation
st.subheader("✍️ Or Enter Your Own Text")
text_input = st.text_area("Enter text to check for safety")

if text_input and st.button("🔍 Moderate Text"):
    result = moderate_text(text_input, tokenizer, llama_guard_model)
    st.warning(f"🛡️ Moderation Result: {result}")
