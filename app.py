# lets import packagaes
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64 
import os

# langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime


# getting text chunks from pdf

# get text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

# get chunks

def get_text_chunks(text, model_name):
    if(model_name=="Google AI"):
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks= text_splitter.split_text(text)
    return chunks 

# embedding chunks and then storing in vector store

def get_vector_store(text_chunks, model_name, api_key= None):
    if model_name=="Google AI":
        embeddings= GoogleGenerativeAIEmbeddings(model="model/embedding-001", google_api_key= api_key )
    vector_store= FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store= save_local("faiss_index")
    return vector_store


# create a conversation using langchain 

def get_conversationl_chain(model_name, vectorstore=None, api_key= None):
    if model_name=="Google AI":
        prompt_template=""" Answer the question as detailed as possible from the provided context, make sure to provide all
            the details with proper structure, if the answer is not in the provided context just say, "answer is not available in the context", don't provide wrong answer \n\n Context:\n {context}? \n
                Question:\n{question}?\n 
            Answer: 
            """
        model= ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,google_api_key= api_key)
        prompt= PromptTemplate(template=prompt_template, input_variables=["context","question"])
        chain= load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain


