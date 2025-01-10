import streamlit as st
from rag import extract_text_from_pdf, helper_graph


PDF_NAME = "uploaded.pdf"


st.title("PDF Tester")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # save the file
    with open(PDF_NAME, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # read the texts and display
    extract_data = extract_text_from_pdf(PDF_NAME)
    # check pdf related to IBD or not
    # execute the graph
    with st.spinner("Analysing............"):
        get_content = helper_graph.invoke({"pdf_path": PDF_NAME})
    # display content
    if get_content["accept_content"]:
        st.subheader("PDF Information")
        st.write(get_content["content"])
    else:
        st.subheader("Ooopss!! Not Related to IBD!")
        st.write(get_content["content"])

