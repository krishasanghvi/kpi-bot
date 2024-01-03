# to run---> streamlit run app.py


# pip install langchain streamlit pypdf2 faiss-cpu openai camelot-py
# ghostscript(path to be added)
""" 
langchain---> framework designed to simplify the creation of applications using large language models (LLMs)

streamlit--->create web apps for data science and machine learning in a little time directly from python code

pypdf2--->A pure-python PDF library capable of splitting, merging, cropping, and transforming PDF files

faiss-cpu--->Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.

"""



from kpi_list import *
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
import re
import os
from io import BytesIO
import camelot
import tabula
from langchain.agents import create_csv_agent
from langchain.document_loaders.csv_loader import CSVLoader
from getpass import getpass

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# extracts info from pdfs and converts them to text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(
        r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text
    )
    return text


# converts text to smaller chunks for easier use
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


# a combined function for the above two functions
def get_and_split_combined(pdf_input):
    raw = get_pdf_text(pdf_input)
    clean = clean_text(raw)
    clean_wt_space = " ".join(clean.split())
    chunks = get_text_chunks(clean_wt_space)
    return chunks


# converts the chunks to embeddings ie convert words to vectors or arrays of numbers that represent the meaning and the context of the tokens that the model processes and generates. Embeddings are derived from the parameters or the weights of the model, and are used to encode and decode the input and output texts.
def embed(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return db


# embeds and stores csv files
def embed_tables(info):
    embeddings = OpenAIEmbeddings()
    dbt = FAISS.from_documents(info, embeddings)
    return dbt


def solving(question, db, final_doc) -> str:
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.4),
        chain_type="map_rerank",
        retriever=db.as_retriever(),
        return_source_documents=True,
    )
    response = qa(question.strip())
    st.write(response)
    final_doc += "\n"
    final_doc += f"output: {response['result']} \n"
    final_doc += f"source:{response['source_documents']} \n"
    return final_doc


# inputs each kpi as a question uses solve function to get the answer and adds the output to final_docs
def kpi_processing(kpi_category, db, template, final_doc) -> str:
    category_len = len(kpi_category)
    for no in range(0, category_len):
        kpi = kpi_category[no]
        final_doc += "Q."
        final_doc += kpi
        final_doc += "\n"
        prompt = PromptTemplate.from_template(template)
        final_doc = solving(prompt.format(kpi=kpi), db, final_doc)
    return final_doc


def filesFromList(file_inputs: List[UploadedFile]) -> list["str"]:
    listss = []
    [listss.append(fileFromBytes(file_input, listss)) for file_input in file_inputs]
    return listss


def fileFromBytes(file_input: UploadedFile, filename_list: List[str]) -> list["str"]:
    val = file_input.getvalue()
    obj = BytesIO(val)
    with open(file_input.name, "wb") as f:
        f.write(obj.getbuffer())
    return file_input.name


def tables_info(tables):
    for idx, table in enumerate(tables, start=1):
        table_csv_path = (
            f"C:\\Users\\sanghvik\\Documents\\python\\multiple_pdfs\\table_{idx}.csv"
        )
        table.to_csv(table_csv_path, index=False)
        loader = CSVLoader(
            file_path=table_csv_path,
            encoding="utf-8",
            csv_args={"delimiter": ","}
        )
        
        info = loader.load()


def main():
    final_doc = ""
    st.set_page_config(page_title="PDF Searchbot")
    st.header("PDF Searchbot")
    question = st.text_input("Ask a question about your documents")
    # type1 KPI's
    type1= st.multiselect(
        "type1 KPI's", type1list, key="eqt"
    )
    # type2 KPI's 
    type2= st.multiselect(
        "type2 KPI's ",
        type2list,
        key="sqt",
    )

    # type3 KPI's
    type3= st.multiselect(
        "type3 KPI's", type3list, key="gqt"
    )


    with st.sidebar:
        st.subheader("Add documents")
        pdfs = st.file_uploader("Add documents", accept_multiple_files=True)

    if st.button("Process"):
        with st.spinner("Processing"):
            if pdfs != []:
                chunks = get_and_split_combined(pdfs)
                db = embed(chunks)
                pdf_name = filesFromList(pdfs)
                for i in range(0, len(pdf_name)):
                    name = pdf_name[i]
                    path = r"C:\Users\sanghvik\Documents\python\multiple_pdfs"
                    pdf_path = f"{path}\{name}"
                    tables = camelot.read_pdf(pdf_path, pages="all")
                    tables1 = tabula.read_pdf(pdf_path, pages="all")
                    info = tables_info(tables)
                    info1 = tables_info(tables1)
                    if info is not None:
                        dbc = embed_tables(info)
                        db.merge_from(dbc)
                    if info1 is not None:
                        dbt = embed_tables(info1)
                        db.merge_from(dbt)
                if question:
                    final_doc = solving(question, db, final_doc)
                template_qt = """{kpi} The question is quantitative. Return the value"""
                template_ql = """{kpi} The question is qualitative."""
                final_doc = kpi_processing(
                    type1, db, template_qt, final_doc
                )
                final_doc = kpi_processing(
                    type2, db, template_q1, final_doc
                )
                final_doc = kpi_processing(
                    type3, db, template_qt, final_doc
                )
                st.download_button("Download file", final_doc)


if __name__ == "__main__":
    main()
