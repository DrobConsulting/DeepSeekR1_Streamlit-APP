# DeepSeek R1 Tuned Chat Application

A chat interface powered by DeepSeek R1 models, using a vLLM backend and LangChain for real-time streaming. Built with Streamlit, this app supports multiple conversation threads, streaming responses, and an extracted "thinking" process.

## Features

- **Supports Any DeepSeek Model** – Easily switch models by updating `MODEL_NAME`.
- **Real-Time Streaming** – Uses LangChain for instant response streaming.
- **Hidden Thought Extraction** – Extracts internal `<think>...</think>` reasoning, displaying it separately.
- **Multi-Conversation Support** – Manage multiple chat threads via a sidebar.
- **Easy Deployment** – Configurable via environment variables, with a simple Streamlit UI.

---

## Prerequisites

- Python 3.8+
- vLLM installed and configured
- Streamlit, LangChain, OpenAI Python Client

---

## Installation

1. **Set up a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**  
   ```bash
   pip install streamlit langchain openai
   ```

---

## Running the vLLM Server

Start vLLM with a DeepSeek model:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
```

Replace `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` with another model if needed.

Ensure the server is running at `http://127.0.0.1:8000/v1`.

---

## Running the Chat Application

Once vLLM is running, start the chat app:

```bash
streamlit run app.py
```

---

## Usage

1. **Start or select a conversation** – Use the sidebar to manage chats.
2. **Send messages** – Responses stream in real-time.
3. **View thought process** – Hidden `<think>...</think>` reasoning is displayed separately.

---

## Customization

Modify `MODEL_NAME` in the code to change models:

```python
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

Adjust temperature and token limits:

```python
TEMPERATURE = 0.3
MAX_TOKENS = 5000
```

---

## Troubleshooting

- **Server not connecting?** Ensure vLLM is running at `http://127.0.0.1:8000/v1`.
- **Missing dependencies?** Reinstall requirements:
  ```bash
  pip install streamlit langchain openai
  ```
- **Streaming issues?** Check vLLM logs for model execution errors.

---

## License

**MIT**

---

## Acknowledgements

- **DeepSeek Team** – For the DeepSeek R1 model and prompting guide.
- **LangChain & Streamlit Communities** – For enabling real-time AI chat apps.
