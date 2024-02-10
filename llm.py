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

# Use this only if you want to run it via Cloudflare Workers AI
def use_cf_workers():
    from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
    from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings

    cf_account_id = os.environ.get("CF_ACCOUNT_ID")
    cf_api_token = os.environ.get("CF_API_TOKEN")

    llm = CloudflareWorkersAI(
        account_id=cf_account_id,
        api_token=cf_api_token,
        model="@cf/mistral/mistral-7b-instruct-v0.1"
    )

    embeddings = CloudflareWorkersAIEmbeddings(
        account_id=cf_account_id,
        api_token=cf_api_token,
        model_name="@cf/baai/bge-large-en-v1.5",
    )

    return llm, embeddings

# use_from_env will use the model specified in the MODEL environment variable
def use_from_env() -> tuple[BaseChatModel, Embeddings]:
    model = os.environ.get("MODEL", "mistral")

    match model:
        case "ollama":
            return use_ollama()
        case "mistral":
            return use_mistral()
        case "cloudflare":
            return use_cf_workers()
        case _:
            raise ValueError(f"Unknown model: {model}")

# Change this to use the model you want
llm, embeddings = use_from_env()
