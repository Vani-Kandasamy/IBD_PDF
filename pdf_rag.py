import PyPDF2
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import  Annotated
import operator
from pinecone import Pinecone
from openai import OpenAI
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
import base64
import os
import streamlit as st

import prompts as pt


TEXT_MODEL = "text-embedding-ada-002"
NAMESPACE_KEY = "Keya"
LLM_MODEL = "gpt-4o-mini"


os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"]= st.secrets["PINECONE_API_KEY"]
os.environ["INDEX_HOST"]= st.secrets["INDEX_HOST"]


# set the openai model
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
# create client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# this is be default has the messages and add_messages reducers
class BotState(MessagesState):
    pdf_path: str
    content: str
    accept_content: bool
    answer: str


class PdfChecker(BaseModel):
    pdf_related: bool = Field(None, description = "True if the pdf is about IBD disease else False")


import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_text = ""
    # read the pdf
    with open(pdf_path, 'rb') as file_:
        reader = PyPDF2.PdfReader(file_)
        # extract the text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()

    final_text = pdf_text.strip().replace("\n", "")

    return final_text


def pdf_data_extractor(state: BotState):
    pdf_path = state["pdf_path"]
    extracted_text = extract_text_from_pdf(pdf_path)

    return {"content": extracted_text}


def ibd_tester(state: BotState):
    extracted_pdf_text = state["content"]

    # create the structured output llm
    structured_llm = llm.with_structured_output(PdfChecker)

    # generate the prompt as a system message
    system_message_prompt = [SystemMessage(pt.PDF_CHECK_INSTRUCTIONS.format(content = extracted_pdf_text))]

    # invoke the llm
    invoke_results = structured_llm.invoke(system_message_prompt)
    print("IBD TESTER", invoke_results.pdf_related)

    return {"accept_content": invoke_results.pdf_related}


def user_guider(state: BotState, config: RunnableConfig):
    # invoke the llm
    invoke_image_query = llm.invoke([SystemMessage(content=pt.GUIDANCE_INSTRUCTIONS)], config)

    return {"answer":invoke_image_query.content }


def conditional_checker(state: BotState):
    content_status = state["accept_content"]

    if content_status:
        return "answer_generator"

    return "guidelines_generator"


def answer_generator(state: BotState, config: RunnableConfig):
    pdf_context = state["content"]
    messages = state["messages"]

    # generate the prompt as a system message
    system_message_prompt = [SystemMessage(pt.ANSWER_INSTRUCTIONS.format(context = pdf_context ))]
    # invoke the llm
    answer = llm.invoke(system_message_prompt + messages, config)

    return {"answer": answer}


# add nodes and edges
helper_builder = StateGraph(BotState)
helper_builder.add_node("pdf_text_extractor", pdf_data_extractor)
helper_builder.add_node("ibd_tester", ibd_tester)
helper_builder.add_node("guidelines_generator", user_guider)
helper_builder.add_node("answer_generator", answer_generator)

# build graph
helper_builder.add_edge(START, "pdf_text_extractor")
helper_builder.add_edge("pdf_text_extractor", "ibd_tester")
helper_builder.add_conditional_edges("ibd_tester", conditional_checker, ["answer_generator", "guidelines_generator"])
helper_builder.add_edge("guidelines_generator", END)
helper_builder.add_edge("answer_generator", END)

# compile the graph
helper_graph = helper_builder.compile()


async def graph_streamer(pdf_path: str, user_query: str):
    # nodes to stream
    # two nodes are there because we are conditionally streaming
    node_to_stream = 'answer_generator'
    other_node_to_stream = 'guidelines_generator'
    # set thread for configuration
    model_config = {"configurable": {"thread_id": "1"}}

    async for event in helper_graph.astream_events({"pdf_path": pdf_path, "messages": user_query}, model_config, version="v2"):
        # Get chat model tokens from a particular node
        #print(event)
        if event["event"] == "on_chat_model_stream":
            if event['metadata'].get('langgraph_node','') == node_to_stream or  event['metadata'].get('langgraph_node','') == other_node_to_stream:
                data = event["data"]
                yield data["chunk"].content