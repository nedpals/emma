from llm import llm
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# this prompt will be used to generate a search query to look up in the document
initial_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "generate a search to look up in order to get information relevant to the conversation")
])

# this prompt will be used for the chatbot to ask the user a question
history_aware_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based only on the below context and make it straight to the point:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

non_history_aware_prompt = ChatPromptTemplate.from_template("""Answer the user's question based only on the below context and make it straight to the point:

{context}
Question: {input}""")

def create_handbook_retrieval_chain(vector: Chroma, history_aware = True):
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
