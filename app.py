import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# Title
st.title("AI Image Generator 🎨")
st.write("Type a prompt and watch AI create an image!")

# Load model (cached so it doesn't reload each time)
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # use NVIDIA GPU
    return pipe

pipe = load_model()

# User input
prompt = st.text_input("Enter your prompt:", "A futuristic city at sunset")

if st.button("Generate"):
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)