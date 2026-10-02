from langchain.chat_models import init_chat_model

from config import LLM_BASE_URL,LLM_API_KEY,LLM_MODEL_ID





model = init_chat_model(
    model = LLM_MODEL_ID,
    api_key = LLM_API_KEY,
    base_url = LLM_BASE_URL,
)

