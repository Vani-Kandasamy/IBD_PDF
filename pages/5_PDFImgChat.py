import streamlit as st
from pdf_rag import graph_streamer
from langchain_core.messages import AIMessage, HumanMessage
import fitz  # PyMuPDF
from PIL import Image
import io
import os

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
    #Extract images from a PDF and save them as PNG files.
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Open the image using Pillow
            image = Image.open(io.BytesIO(image_bytes))

            # Convert CMYK images to RGB
            if image.mode == "CMYK":
                image = image.convert("RGB")

            image_path = f"images/page_{page_number + 1}_image_{image_index + 1}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            images.append(image_path)

    pdf_document.close()
    return images

# Initialize session state messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("PDF Chatter")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", on_change=empty_message_list)

if uploaded_file:
    # Save the PDF
    with open(PDF_NAME, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract and display images
    image_paths = extract_images_from_pdf(PDF_NAME)
    if image_paths:
        st.subheader("Extracted Images")
        for image_path in image_paths:
            st.image(image_path, use_column_width=True)
    else:
        st.write("No images found in the PDF.")

    # Display chat messages
    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.container():  # Using a container here instead of st.chat_message
            st.write(f"**{message['role'].capitalize()}:** {message['content']}")

    # Input text for chat
    input_text = st.text_input("You: ", "")
    if st.button("Send") and input_text.strip() != "":
        st.session_state.messages.append({"role": "user", "content": input_text})

        with st.container():
            st.write(f"**User:** {input_text}")

        with st.spinner("Generating response..."):
            message_list = message_creator(st.session_state.messages)
            response = graph_streamer(PDF_NAME, message_list)  # Assuming this function interacts with the PDF
            st.session_state.messages.append({"role": "assistant", "content": response})

        with st.container():
            st.write(f"**Assistant:** {response}")