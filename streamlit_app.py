import pandas as pd
import streamlit as st
import warnings
import sys
import io
from tabulate import tabulate
from classes import get_primer, format_question, run_request, get_text_primer
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="DocSense")

st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:'Trebuchet MS', sans-serif; padding-top: 0rem;'> \
            DocSense</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>by Adarsh Agarwal, Saakar Chaudhary & Farhan Shaikh</h2>", unsafe_allow_html=True)

# Sidebar elements

# ... (Other sidebar elements remain unchanged)

if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("movies.csv")
    datasets["Cars"] =pd.read_csv("cars.csv")
    datasets["Colleges"] =pd.read_csv("colleges.csv")
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

hf_key = st.text_input(label = ":hugging_face: HuggingFace Key:", help="Required for Code Llama", type="password")

# ... (The part where you handle datasets remains unchanged)
with st.sidebar:
    mode = st.radio("Select Mode:", ("Visualize", "Chat"))
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset
    try:
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
        index_no=0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets. Give it a nice name.
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            # We want to default the radio button to the newly added dataset
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:",datasets.keys(),index=index_no)#,horizontal=True,)

if mode == "Visualize":
  question = st.text_area(":eyes: What would you like to visualise?", height=10)
  go_btn = st.button("Go...")
  
  # Execute chatbot query
  if go_btn:
      if not hf_key.startswith('hf_'):
          st.error("Please enter a valid HuggingFace API key.")
      else:
          st.subheader("Code Llama")
          try:
              # Get the primer for this dataset
              primer1, primer2 = get_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]') 
              # Format the question 
              question_to_ask = format_question(primer1, primer2, question, "Code Llama")   
              # Run the question
              answer = run_request(question_to_ask, "CodeLlama-34b-Instruct-hf", alt_key=hf_key)
              # Add to the beginning of the script
              answer = primer2 + answer
              plot_area = st.empty()
              plot_area.pyplot(exec(answer))           
          except Exception as e:
              st.error(f"An error occurred: {e}")

elif mode == "Chat":
            chat_question = st.text_area("What question do you have about the dataset?", height=10)
            chat_btn = st.button("Ask")
            
            if chat_btn:
                try:
                    output_dict={}
                    primer1, primer2 = get_text_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]')
                    question_to_ask = format_question(primer1, primer2, chat_question, "Code Llama")
                    answer = run_request(question_to_ask, "CodeLlama-34b-Instruct-hf", alt_key=hf_key)
                    answer = primer2 + answer
                    output_dict['datasets'] = datasets
                    st.write(answer)
                    exec(answer, output_dict)
                    answer = result()
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
# ... (The part where you display the datasets in tabs and add the footer remains unchanged)
tab_list = st.tabs(datasets.keys())

# Load up each tab with a dataset
for dataset_num, tab in enumerate(tab_list):
    with tab:
        # Can't get the name of the tab! Can't index key list. So convert to list and index
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name],hide_index=True)

# Insert footer to reference dataset origin  
footer="""<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
<p> <a style='display: block; text-align: center;'> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
st.caption("Datasets courtesy of NL4DV, nvBench and ADVISor")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
