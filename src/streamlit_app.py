# ./src/streamlit_app.py
#
# Main steamlit app.

# Streamlit import.
import streamlit as st

# Env import.
from os import getenv
from dotenv import load_dotenv

# LangChain imports.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Pydantic and native enum imports.
from pydantic import BaseModel, Field
from enum import Enum

# ------------------------------------------------------------------------------
# Loads the OPENAI_API_KEY from the .env file.
# Loads the PINECONE_API_KEY from the .env file.
load_dotenv()

# Retrieve the API keys from environment variables
openai_api_key = getenv('OPENAI_API_KEY')
pinecone_api_key = getenv('PINECONE_API_KEY')

# ------------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

index_name = "hackohio2024"

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ------------------------------------------------------------------------------
# Initialize the language model (with structured output handling)
mini_llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOpenAI(model="gpt-4o")

class Category(str, Enum):
    GENERAL = "general"
    SPECIFIC = "specific"

class CheckQueryRelevance(BaseModel):
    verdict: bool = Field(description="""
                          Determin true or false that the query is related to AI research and computer science.
                          Example 1: "What is the best way to train a neural network?" -> True
                          Example 2: "What is the best way to train my dog?" -> False
                          """)
class CategorizeQuery(BaseModel):
    category: Category = Field(description="""
                             Determin "general" or "specific" that the query is about AI research and computer science.
                             Example 1: "What is the best way to train a neural network?" -> "specific"
                             Example 2: "How do I write a research paper?" -> "general"
                             """)

# Structured output setup to check if the query is research-related
relevance_llm = mini_llm.with_structured_output(CheckQueryRelevance)
category_llm = mini_llm.with_structured_output(CategorizeQuery)

# Define the get_response function
def get_response(query, chat_history):

    is_research_related = relevance_llm.invoke(query).verdict
    if is_research_related:
        query_category = category_llm.invoke(query).category
        if query_category == Category.GENERAL:
            return get_general_stream(query, chat_history)
        else:
            return get_specific_stream(query, chat_history)
    else:
        return "Please ask me questions related to OSU CSE research, and I'd be happy to help."    

def get_general_stream(query, chat_history):
    template = """
    You are a undergrad AI and computer science research advisor here to help university students
    with their general questions about AI and computer science research.

    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query,
    })

def get_specific_stream(query, chat_history):

    results = vector_store.similarity_search(
        query,
        k=1,
        filter={},
    )
    best_match_faculty_info = results[0].page_content

    template = """
    You are a undergrad AI and computer science research advisor here to help university students
    with their general questions about AI and computer science research.
    You are given the bio of the best-matching faculty member based on the user's question.

    Summarize the faculty member's bio.

    Put a horizontal line here.
    
    Suggest the best course of action for the student based on the faculty member's bio.
    
    Best match faculty info: {best_match_faculty_info}

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "best_match_faculty_info": best_match_faculty_info,
        "chat_history": chat_history,
        "user_question": query,
    })

# ------------------------------------------------------------------------------
# Prompts used for testing.
# I want to get into auto ai reserach at OSU, help me draft a email to the most famous faculty on this subject.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Init the chat history.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------------------------
# Web app page setup.
st.set_page_config(page_title="🌰 Buck-AI-Guide")
st.title("🌰 Buck-AI-Guide")
st.info("Welcome to Buck-AI-Guide! I’m here to help you explore AI research at OSU by connecting you with professors or answering general AI concepts. Ask me anything!")

# Print the converstation history.
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
             st.markdown(message.content)

# Get the latest user question and process the response.
user_query = st.chat_input("your message here")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        hybrid_response = get_response(user_query, st.session_state.chat_history)
        if isinstance(hybrid_response, str):
            ai_response = hybrid_response
            st.write(hybrid_response)
        else:
            ai_response = st.write_stream(hybrid_response)

    st.session_state.chat_history.append(AIMessage(ai_response))
