import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Use this only if you want to run it locally via Ollama
def use_ollama():
    from langchain.llms.ollama import Ollama
    from langchain_community.embeddings.ollama import OllamaEmbeddings

    llm = Ollama(name="llama2", temperature=0)
    embeddings = OllamaEmbeddings()

    return llm, embeddings

# Use this only if you want to run it via Mistral AI
def use_mistral():
    from langchain_mistralai.chat_models import ChatMistralAI
    from langchain_mistralai.embeddings import MistralAIEmbeddings

    api_key = os.environ.get("MISTRAL_API_KEY")

    llm = ChatMistralAI(
        model="mistral-small",
        mistral_api_key=api_key,
        max_tokens=4096,
        safe_mode=True)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key)

    return llm, embeddings

# use_from_env will use the model specified in the MODEL environment variable
def use_from_env() -> tuple[BaseChatModel, Embeddings]:
    model = os.environ.get("MODEL", "mistral")

    match model:
        case "ollama":
            return use_ollama()
        case "mistral":
            return use_mistral()
        case _:
            raise ValueError(f"Unknown model: {model}")

# Change this to use the model you want
llm, embeddings = use_from_env()
