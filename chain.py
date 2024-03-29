from llm import llm
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import meta

# this prompt will be used to generate a search query to look up in the document
initial_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "generate a search to look up in order to get information relevant to the conversation")
])

# this prompt will be used to welcome the user
welcome_prompt = f"Your name is {meta.title} and you are a chatbot and you are {meta.full_description}"

# this will be the base prompt for the chatbot to ask the user a question both for history aware and non-history aware
base_system_prompt = """Answer the user's question based only on the below context. Anything that refers to their school is also your school. Make it straight to the point and without using phrases like 'based on the provided context', 'according to the information given', 'is not provided in the given context'. If you don't know the answer, just say that you don't know, don't try to make up an answer. Here's the context:

{context}"""

# this prompt will be used for the chatbot to ask the user a question
history_aware_prompt = ChatPromptTemplate.from_messages([
    ("system", welcome_prompt + base_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# this prompt will be used if it does not have access to the chat history
non_history_aware_prompt = ChatPromptTemplate.from_template(
    welcome_prompt + base_system_prompt + "Question: {input}"
)

def create_handbook_retrieval_chain(vector: VectorStore, history_aware = True):
    retriever = vector.as_retriever()
    if history_aware:
        retriever_chain = create_history_aware_retriever(llm, retriever, initial_prompt)
        document_chain = create_stuff_documents_chain(llm, history_aware_prompt)

        # use retriever_chain to be able to use the chat history
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    else:
        document_chain = create_stuff_documents_chain(llm, non_history_aware_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
