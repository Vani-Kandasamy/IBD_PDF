import streamlit as st
from rag import extract_text_from_pdf


PDF_NAME = "uploaded.pdf"


st.title("PDF Reader")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # save the file
    with open(PDF_NAME, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # read the texts and display
    extract_data = extract_text_from_pdf(PDF_NAME)
    # disply the pdf
    st.subheader("Extracted Data")
    st.markdown(extract_data)