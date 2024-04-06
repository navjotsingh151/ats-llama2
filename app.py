import streamlit as st
# import google.generativeai as genai
import os
import PyPDF2 as pdf
import pyperclip
import json
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

def generate_prompt(repo_id, text, jd):

    hub_llm = HuggingFaceHub(repo_id=repo_id)

    input_prompt="""
    Hey Act Like a skilled or very experience ATS(Application Tracking System)
    with a deep understanding of tech field,software engineering,data science ,data analyst
    and big data engineer. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide 
    best assistance for improving thr resumes. Assign the percentage Matching based 
    on Jd and
    the missing keywords with high accuracy
    resume:{text}
    description:{jd}

    Generate detailed Cover Letter based on description and resume in 500 words
    {{"Cover Letter":""}}
    """
    prompt = PromptTemplate(input_variables=["text", "jd"], template=input_prompt)

    hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
    return hub_chain.run(text=text,
                          jd=jd)

def llama_response(repo_id, text, jd):

    hub_llm = HuggingFaceHub(repo_id=repo_id)

    input_prompt="""
    Hey Act Like a skilled or very experience ATS(Application Tracking System)
    with a deep understanding of tech field,software engineering,data science ,data analyst
    and big data engineer. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide 
    best assistance for improving thr resumes. Assign the percentage Matching based 
    on Jd and
    the missing keywords with high accuracy
    resume:{text}
    description:{jd}

    I want the response in one single string having the structure
    {{"JD Match":"%","MissingKeywords:[]","create 2 job duties/responsibilities fromm missing keywords:":""}}
    """
    prompt = PromptTemplate(input_variables=["text", "jd"], template=input_prompt)

    hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=False)
    return hub_chain.run(text=text,
                          jd=jd)

def clean_response(response, matching_string):
    # Find the index of the matching string in the response
    start_index = response.find(matching_string)
    if start_index != -1:
        # Extract the substring from the start index to the end
        cleaned_response = response[start_index:]
        return cleaned_response
    else:
        return "Matching string not found in response."
    

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

def display_response(response):
    # Define CSS styles for key and value highlighting
    key_style = "color: #077b8a;"  # Gold colorf
    value_style = "color: #e75874;"  # Lime green color
    
    # Split the response into lines
    lines = response.split(", \"")
    print(lines)
    
    # Iterate through each line of the response
    for line in lines:
        # Highlight keys and values with different colors
        line = line.replace('{"', '<span style="' + key_style + '">{"') \
                   .replace('":', '"</span>":<span style="' + value_style + '">') \
                   .replace(',"', '</span>,"<span style="' + key_style + '">') \
                   .replace('"}', '"}</span>')
        # Display the line with Markdown formatting
        st.markdown(line, unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #000000;  /* Black background color */
        color: #FFFFFF;  /* White text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)




## streamlit app

st.title("Smart ATS")
st.text("Improve Your Resume ATS")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")

col1, col2 = st.columns(2)
# col1.width, col2.width = 500, 500

# Add buttons to the columns
with col1:

    submit = st.button("Submit")

with col2:
    generate_prompt_for_gpt = st.button("Generate Cover Letter")

if submit:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        st.subheader("Response:")
        response = llama_response(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                                  text=text,
                                  jd=jd)
        response_clean = clean_response(response=response,
                                        matching_string="""{\"JD Match": \"""")
        # display_response(response_clean)
        st.code(response_clean, language='json')


if generate_prompt_for_gpt:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        st.subheader("Response:")
        response = generate_prompt(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                                   text=text,
                                  jd=jd)
        
        response_clean = clean_response(response=response,
                                        matching_string="""\"Cover Letter\":\"""")
        st.text(response_clean)
