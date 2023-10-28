import streamlit as st
import pandas as pd
from classes import run_request  # Assuming run_request comes from the 'classes' module
from langchain import HuggingFaceHub, LLMChain, PromptTemplate

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Data App')
st.title('🦜🔗 Ask the Data App')

# Load CSV file
def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

# Generate LLM response
def generate_response(csv_file, input_query, hf_key):
    llm = HuggingFaceHub(huggingfacehub_api_token=hf_key, repo_id="TheBloke/Llama-2-70B-chat-GPTQ", model_kwargs={"temperature": 0.1, "max_new_tokens": 500})
    df = load_csv(csv_file)
    prompt = f"""Your task is to answer the following question based on the dataframe `df`:
    {input_query}"""
    
    llm_prompt = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
    response = llm_chain.predict()
    return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
    'How many rows are there?',
    'What is the range of values for MolWt with logS greater than 0?',
    'How many rows have MolLogP value greater than 0.',
    'Other']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
hf_key = st.text_input('HuggingFace API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text == 'Other':
    query_text = st.text_input('Enter your query:', placeholder='Enter query here ...', disabled=not uploaded_file)
if not hf_key.startswith('hf_'):
    st.warning('Please enter your HuggingFace API key!', icon='⚠')
if hf_key.startswith('hf_') and (uploaded_file is not None):
    st.header('Output')
    generate_response(uploaded_file, query_text, hf_key)
