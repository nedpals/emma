from providers.lm_studio import LMStudioProvider

provider = LMStudioProvider(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    llm_model="gemma-4-E4B-it",
    vlm_model="gemma-4-E4B-it",
    embedding_model="text-embedding-nomic-embed-text-v1.5",
)
