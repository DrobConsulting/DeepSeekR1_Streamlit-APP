import os
import logging
import sys
import re
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import LLMResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="DeepSeek R1 Tuned Chat", layout="wide")

# ---- Configuration ----
VLLM_SERVER_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TEMPERATURE = 0.3
MAX_TOKENS = 5000

os.environ["OPENAI_API_BASE"] = VLLM_SERVER_URL
os.environ["OPENAI_API_KEY"] = "dummy"

logger.info("Environment variables set successfully.")

# ---- Initialize state for multiple conversations ----
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {"Default": []}

if "thinking_history" not in st.session_state:
    st.session_state["thinking_history"] = {}

# Sidebar for selecting or creating a new conversation
st.sidebar.title("Conversations")
conversation_names = list(st.session_state["conversations"].keys())
selected_conversation = st.sidebar.selectbox("Select a conversation:", conversation_names, index=0)

# Option to add a new conversation
new_convo_name = st.sidebar.text_input("Create new conversation:")
if st.sidebar.button("Add Conversation") and new_convo_name.strip():
    if new_convo_name not in st.session_state["conversations"]:
        st.session_state["conversations"][new_convo_name] = []
        selected_conversation = new_convo_name  # Switch to the new conversation
        st.rerun()

# Ensure the current session is tracked
if selected_conversation not in st.session_state["conversations"]:
    st.session_state["conversations"][selected_conversation] = []

messages = st.session_state["conversations"][selected_conversation]  # Active conversation messages

# ---- Simple parser for <think>...</think> ----
def parse_think(full_text: str):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, full_text, flags=re.DOTALL)
    if match:
        thinking_part = match.group(1).strip()
        final_text = full_text.replace(match.group(0), "").strip()
        return thinking_part, final_text
    else:
        return "", full_text  # No <think></think> found

# ---- Title & layout ----
st.title(f"Chat with DeepSeek R1 Agent ({selected_conversation})")

# ---- Display conversation history ----
for idx, msg in enumerate(messages):
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Assistant:** {msg.content}")

        # Retrieve and display thought process in an expander
        thinking_part = st.session_state["thinking_history"].get((selected_conversation, idx), None)
        if thinking_part:
            with st.expander(f"🔍 Show Thought Process (Response {idx})", expanded=False):
                st.markdown(thinking_part)

# ---- Input box pinned at the bottom ----
with st.container():
    user_input = st.text_area("Type your message here:", height=150)
    send_button = st.button("Send", type="primary")

# Define assistant placeholder for streaming response
assistant_placeholder = st.empty()

# Streamlit callback handler for streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text_so_far = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text_so_far += token
        self.placeholder.markdown(self.text_so_far)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.placeholder.markdown("**(streaming complete)**")

if send_button and user_input.strip():
    messages.append(HumanMessage(content=user_input))

    # Prepare streaming callback
    stream_handler = StreamlitCallbackHandler(assistant_placeholder)
    callback_manager = CallbackManager([stream_handler])

    # Create the ChatOpenAI instance
    try:
        chat = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            streaming=True,
            callback_manager=callback_manager,
            verbose=True,
        )
        logger.info("ChatOpenAI model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        messages.append(AIMessage(content=f"Error initializing LLM: {str(e)}"))
        st.stop()

    # Get the final response (model streams tokens to assistant_placeholder)
    try:
        response = chat(messages)
    except Exception as e:
        response_text = f"Error: {str(e)}"
        logger.error(f"LLM error: {e}")
        messages.append(AIMessage(content=response_text))
        st.stop()

    # Parse out <think>...</think>
    thinking_part, final_answer = parse_think(response.content)

    # Store thinking process in session state using a (conversation, index) key
    response_index = len(messages)
    st.session_state["thinking_history"][(selected_conversation, response_index)] = thinking_part

    # Store final assistant response in conversation history
    messages.append(AIMessage(content=final_answer))

    # Update session state with the modified conversation
    st.session_state["conversations"][selected_conversation] = messages

    # Re-run so the conversation display updates
    st.rerun()
