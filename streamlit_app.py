import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import transformers
import torch
import gradio as gr
from huggingface_hub import InferenceClient

# Create a Streamlit app
st.title("Summarize a YouTube Video")

# Helper function to get YouTube transcript
def get_yt_transcript(url):
    text = ''
    vid_id = YouTube(url).stream_list[0].video_id
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid_id)
        text = ' '.join([entry['text'] for entry in transcript])
    except:
        pass
    return text

# Helper function to transcribe a YouTube video using the Whisper model
def transcribe_yt_vid(url):
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(filename="audio.mp3")
    
    asr = transformers.pipeline(
        "automatic-speech-recognition",
        model="tiiuae/falcon-7b-instruct",
        device=0,
    )

    asr.model.config.forced_decoder_ids = (
        asr.tokenizer.get_decoder_prompt_ids(
            language="en",
            task="transcribe"
        )
    )

    temp = asr(out_file, chunk_length_s=20)
    text = temp[0]['text']
    
    del(asr)
    torch.cuda.empty_cache()
    
    return text

# Helper function to transcribe a YouTube video using the Hugging Face Hub API
def transcribe_yt_vid_api(url, api_token):
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(filename="audio.wav")
    
    client = InferenceClient(model="openai/whisper-large", token=api_token)

    import librosa
    import soundfile as sf

    text = ''
    t = 25
    x, sr = librosa.load(out_file, sr=None)
    
    for i in range(0, (len(x) // (t * sr)) + 1):
        y = x[t * sr * i: t * sr * (i + 1)]
        split_path = "audio_split.wav"
        sf.write(split_path, y, sr)
        text += client.automatic_speech_recognition(split_path)

    return text

# Helper function to summarize text
def summarize_text(title, text, temperature, words, api_token, do_sample):
    # Your summarization logic here
    return "Summary not implemented yet"

# Streamlit app UI components
url = st.text_input("Enter YouTube video URL:")
api_token = st.text_input("Paste your Hugging Face API token (Optional):")
force_transcribe = st.checkbox("Transcribe even if transcription is available")
do_sample = st.checkbox("Set the Temperature")
temperature = st.slider("Generation temperature", 0.01, 1.0, 0.25)
words = st.slider("Length of the summary", 100, 500, 100)
submit_button = st.button("Summarize!")

if submit_button:
    st.write("Please wait while we summarize the video...")

    # Check if to transcribe the video
    text = ""
    if force_transcribe or not get_yt_transcript(url):
        text = transcribe_yt_vid_api(url, api_token) if api_token else transcribe_yt_vid(url)
        transcript_source = 'The transcript was generated using the Whisper model.'

    # Summarize the video
    title = get_youtube_title(url)
    summary, summary_source = summarize_text(title, text, temperature, words, api_token, do_sample)
    transcript = text
    st.write("Video Title:", title)
    st.write("Summary:", summary)
    st.write("Transcript:", text)
    st.write("Summary Source:", summary_source)
    st.write("Transcript Source:", transcript_source)
