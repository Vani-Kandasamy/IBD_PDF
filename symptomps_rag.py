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

import ibd_prompts as ibd_pt


# constants
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

# pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["INDEX_HOST"])


# to check user query greeting or not
class QueryClassifier(BaseModel):
    symptoms: bool = Field(None, description = "If the user query is about IBD Symptom Inquiry, set it as True. If the user requests Factual Information, set this as False.")


# this is be default has the messages and add_messages reducers
class BotState(MessagesState):
    context: Annotated[list, operator.add]
    answer: str
    query_content: bool


def get_openai_embeddings(text: str) -> list[float]:
    response = client.embeddings.create(input=f"{text}", model=TEXT_MODEL)

    return response.data[0].embedding


# function query similar chunks
def query_response(query_embedding, k = 2, namespace_ = NAMESPACE_KEY):
    query_response = index.query(
        namespace=namespace_,
        vector=query_embedding,
        top_k=k,
        include_values=False,
        include_metadata=True,
    )

    return query_response


def content_extractor(similar_data):
    top_values = similar_data["matches"]
    # get the text out
    text_content = [sub_content["metadata"]["text"] for sub_content in top_values]
    return " ".join(text_content)


def get_similar_context(question: str):
    # get the query embeddings
    quer_embed_data = get_openai_embeddings(question)

    # query the similar chunks
    similar_chunks = query_response(quer_embed_data)

    # extract the similar text data
    similar_content = content_extractor(similar_chunks)

    return similar_content


def classify_user_query(state: BotState):
    user_query = state["messages"]

    # generate the prompt as a system message
    system_message_prompt = [SystemMessage(ibd_pt.CLASSIFY_INSTRUCTIONS)]

    # create a structured output
    structured_llm = llm.with_structured_output(QueryClassifier)
    # invoke the llm to generatr an query
    invoke_structured_query = structured_llm.invoke(system_message_prompt + user_query)

    return {"query_content": invoke_structured_query.symptoms}


def semantic_search(state: BotState):
    question = state["messages"]

    # get the most similar context
    similar_context = get_similar_context(question)

    return {"context": [similar_context]}


def answer_generator(state: BotState, config: RunnableConfig):
    searched_context = state["context"]
    messages = state["messages"]

    # generate the prompt as a system message
    system_message_prompt = [SystemMessage(ibd_pt.ANSWER_INSTRUCTIONS.format(context = searched_context ))]
    # invoke the llm
    answer = llm.invoke(system_message_prompt + messages, config)

    return {"answer": answer}


def symptoms_generator(state: BotState, config: RunnableConfig):
    user_symptoms = state["messages"]

    # generate the prompt as a system message
    system_message_symptomp_prompt = [SystemMessage(ibd_pt.SYMPTOMS_GENERATION_INSTRUCTIONS)]
    # invoke the llm
    answer = llm.invoke(system_message_symptomp_prompt + user_symptoms, config)

    return {"answer": answer}


def conditional_checker(state: BotState):
    content_status = state["query_content"]

    if content_status:
        return "question_generator"

    return "pinecone_retriever"


# add nodes and edges
helper_builder = StateGraph(BotState)
helper_builder.add_node("query_classifier", classify_user_query)
helper_builder.add_node("pinecone_retriever", semantic_search)
helper_builder.add_node("answer_generator", answer_generator)
helper_builder.add_node("question_generator", symptoms_generator)

# build graph
helper_builder.add_edge(START, "query_classifier")
helper_builder.add_conditional_edges("query_classifier", conditional_checker, ["question_generator", "pinecone_retriever"])
helper_builder.add_edge("pinecone_retriever", "answer_generator")
helper_builder.add_edge("question_generator", END)
helper_builder.add_edge("answer_generator", END)

# compile the graph
memory = MemorySaver()
helper_graph = helper_builder.compile()


async def ibd_full_graph_streamer(user_query):
    # nodes to stream
    # two nodes are there because we are conditionally streaming
    node_to_stream = 'answer_generator'
    other_node_to_stream = 'question_generator'
    # set thread for configuration
    model_config = {"configurable": {"thread_id": "1"}}

    async for event in helper_graph.astream_events({"messages": user_query}, model_config, version="v2"):
        # Get chat model tokens from a particular node
        #print(event)
        if event["event"] == "on_chat_model_stream":
            if event['metadata'].get('langgraph_node','') == node_to_stream or  event['metadata'].get('langgraph_node','') == other_node_to_stream:
                data = event["data"]
                yield data["chunk"].content