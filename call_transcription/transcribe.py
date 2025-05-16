import asyncio

import streamlit as st
import tempfile
import openai

import logging

logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("******* Application has started ******")

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Preferably use st.secrets or environment variables for production

st.title("🎙️ Audio Transcription with OpenAI Whisper")
st.write("Upload an `.mp3` file and get the transcript using OpenAI Whisper API.")

# File uploader for MP3
uploaded_file = st.file_uploader("Upload an audio file (.mp3)", type=["mp3"])

# model_option = st.selectbox(
#     "Choose a value:",
#     options=["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
# )

client = openai.AsyncOpenAI(api_key=openai.api_key)

async def transcribe_call():
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # transcription_model = model_option
    with open(tmp_file_path, "rb") as audio_file:
        transcription = await client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
        )
    st.success("Transcription complete!")
    st.text_area("Transcript", transcription.text, height=300)

if uploaded_file:
    # Display file details
    st.audio(uploaded_file, format="audio/mp3")

    st.info("Transcribing audio...")
    try:
        asyncio.run(transcribe_call())
    except Exception as e:
        st.error(f"Error: {e}")
