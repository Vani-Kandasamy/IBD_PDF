import streamlit as st
from pdf_rag import graph_streamer
from langchain_core.messages import AIMessage, HumanMessage
import fitz  # PyMuPDF
from PIL import Image
import io

PDF_NAME = "pdf_chat_uploaded.pdf"


def message_creator(list_of_messages: list) -> list:
    prompt_messages = []
    for message in list_of_messages:
        if message["role"] == "user":
            prompt_messages.append(HumanMessage(content=message["content"]))
        else:
            prompt_messages.append(AIMessage(content=message["content"]))

    return prompt_messages


def empty_message_list():
    st.session_state.messages = []


def extract_images_from_pdf(pdf_path):
    """ Extract images from a PDF and save them as PNG files. """
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_path = f"page_{page_number+1}_image_{image_index+1}.png"
            images.append(image_path)
            image.save(image_path)

    pdf_document.close()
    return images


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("PDF Chatter")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", on_change=empty_message_list)

if uploaded_file:
    # save the pdf
    with open(PDF_NAME, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract and display images
    image_paths = extract_images_from_pdf(PDF_NAME)
    st.subheader("Extracted Images")
    for image_path in image_paths:
        st.image(image_path, use_column_width=True)

    # Handle chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_list = message_creator(st.session_state.messages)
            response = graph_streamer(PDF_NAME, message_list)  # Assuming this is the desired behavior
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})