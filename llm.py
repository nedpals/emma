# from langchain.llms.ollama import Ollama
# from langchain_community.embeddings.ollama import OllamaEmbeddings

# llm = Ollama(name="llama2", temperature=0)
# embeddings = OllamaEmbeddings()

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
import os

api_key = os.environ["MISTRAL_API_KEY"]

llm = ChatMistralAI(
    model="mistral-small",
    mistral_api_key=api_key,
    max_tokens=4096)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=api_key)
