import lmstudio as lms

from typing import BinaryIO, Literal

from openai import OpenAI

from models import Message

# Derived from the LM Studio server endpoint.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_model = "gemma-3-4b-it-qat"
vlm_model = "gemma-3-12b-it-qat"
embedding_model = "text-embedding-nomic-embed-text-v1.5"

def generate_response(prompt: str | list[dict] | list[Message], temperature: float = 0.7, max_tokens: int = -1) -> str:
    # Generate a response from the model using chat completions
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        if all(isinstance(m, dict) for m in prompt):
            messages = prompt
        elif all(isinstance(m, Message) for m in prompt):
            messages = [dict(m) for m in prompt]
        else:
            raise TypeError("Unsupported prompt type. Expected list[dict] or list[Message].")
    else:
        raise TypeError("Unsupported prompt type. Expected str or list[dict].")

    completion_params = {
        "model": llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens > 0:
        completion_params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**completion_params)

    # Extract the generated text from the response
    generated_text = response.choices[0].message.content.strip()
    return generated_text

def get_vision_response(img_data: str | BinaryIO | bytes, prompt: str, temperature = 0.1, response_format = None):
    with lms.Client() as client:
        # Assuming the client has a method to get vision response
        image_handle = client.files.prepare_image(img_data)
        model = client.llm.model(vlm_model)
        chat = lms.Chat()
        chat.add_user_message(content=prompt, images=[image_handle])
        prediction = model.respond(chat, response_format=response_format, config={"temperature": temperature})
        if response_format is not None:
            return prediction.parsed
        return prediction.content

def get_embedding(text: str, purpose: Literal['search_query', 'search_document']):
    # Generate an embedding for the input text
    response = client.embeddings.create(
        model=embedding_model,
        input=[f'{purpose}: {text}'],
    )
    
    # Extract the generated embedding from the response
    embedding = response.data[0].embedding
    return embedding
