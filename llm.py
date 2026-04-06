from providers.lm_studio import LMStudioProvider

provider = LMStudioProvider(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    llm_model="gemma-3-4b-it-qat",
    vlm_model="gemma-3-12b-it-qat",
    embedding_model="text-embedding-nomic-embed-text-v1.5",
)
