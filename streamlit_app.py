import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os

# Load Key
OPENAI_API_KEY = st.secrets.OPENAI_API_KEY

st.title('🦜🔗 Youtube Summarizer')

def generate_response(input_text):
  llm = OpenAI(model_name="gpt3", temperature=0.7, openai_api_key=OPENAI_API_KEY)
  st.info(llm(input_text))

def generate_transcript(url):
    st.video(url) 
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    result = loader.load()

    st.info(generate_summary(result))

def generate_summary(transcript):
    # For short videos
    llm = OpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY)
    
    with get_openai_callback() as cb:
      chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
      summarized = chain.run(transcript)

    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    
    return summarized

with st.form('my_form'):
  text = st.text_area('Enter Youtube Link:')
  submitted = st.form_submit_button('Generate Summary')
  if not OPENAI_API_KEY.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and OPENAI_API_KEY.startswith('sk-'):
    generate_transcript(text)