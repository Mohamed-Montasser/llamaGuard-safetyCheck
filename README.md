# 🖼️ Image Captioning + 🛡️ Moderation App

This is a **Streamlit web app** that performs:

1. **Image Captioning** using BLIP from Hugging Face 🤖  
2. **Content Moderation** using [LLaMA Guard 2](https://huggingface.co/meta-llama/LlamaGuard-2-8b) via [Together.ai](https://www.together.ai/) 🔒  
3. **Caption Verification** using a fine-tuned DistilBERT ONNX model from Hugging Face 🎯

---

## 🚀 Features

- Upload any image and automatically generate a caption.
- Moderate the generated caption using LLaMA Guard 2.
- Perform a second-layer moderation using a custom DistilBERT ONNX model.
- Optionally enter your own text to check its safety.
- Lightweight and fast with ONNX + Streamlit.

---

## 🧠 Tech Stack

- `Streamlit` – UI
- `transformers` – BLIP & Tokenizer
- `Together.ai` – LLaMA Guard moderation
- `ONNXRuntime` – for fast DistilBERT inference
- `huggingface_hub` – downloading ONNX model and label encoder
- `joblib` – label encoder loading
- `PIL` – image handling

---

## 🔑 Model Info

- **Captioning Model**: [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **Moderation Model**: `meta-llama/LLaMA Guard-2-8b` via [Together.ai](https://www.together.ai/)
- **Verification Model**: [`M-Montasser/finetuned-distilbert`](https://huggingface.co/M-Montasser/finetuned-distilbert) (ONNX)

This custom DistilBERT model checks whether the caption is **safe** or **unsafe** using ONNX for fast CPU/GPU inference.

---

## 🌐 Live Demo

Check out the app live here: [🔗 llamaguard-safetycheck.streamlit.app](https://llamaguard-safetycheck.streamlit.app/)

## ✍️ Author

**Mohamed Montasser**  
[@M-Montasser](https://huggingface.co/M-Montasser)
