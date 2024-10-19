# ./src/streamlit_app.py
#
# Main steamlit app.

# Streamlit import.
import streamlit as st

# Env import.
from dotenv import load_dotenv

# LangChain imports.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Pydantic imports.
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Loads the OpenAI API key into env vars.
load_dotenv()

# ------------------------------------------------------------------------------
# Initialize the language model (with structured output handling)
llm = ChatOpenAI(model="gpt-4o-mini")

class CheckIsResearchQuestion(BaseModel):
    verdict: bool = Field(description="""
                          Whether the question is related university research or not. 
                          Example 1, true for asking about a department.
                          Example 2, false for asking about admission.
                          """)

# Structured output setup to check if the query is research-related
structured_llm = llm.with_structured_output(CheckIsResearchQuestion)

# Define the get_response function
def get_response(query, chat_history):
    # Step 1: Check if the query is research-related.
    # If the verdict is False, short-cicuit and return the static message.
    structured_output = structured_llm.invoke(query)
    if not structured_output.verdict:
        return "Please ask me questions related to OSU CSE research, and I'd be happy to help."
    
    # Step 2: If the query is research-related, proceed with the full response.
    template = """
    You are a undergrad research advisor here to help university students
    with their questions about research.

    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI()
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query,
    })

# ------------------------------------------------------------------------------
# Init the chat history.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------------------------
# Web app page setup.
st.set_page_config(page_title="Buck-AI-Guide")
st.title("Buck-AI-Guide")

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
