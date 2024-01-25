from llm import llm, embeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

vector = Chroma(persist_directory="./embeddings_db", embedding_function=embeddings)

# this prompt will be used to generate a search query to look up in the document
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "generate a search to look up in order to get information relevant to the conversation")

])

retriever = vector.as_retriever()
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# this prompt will be used for the chatbot to ask the user a question
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based only on the below context and make it straight to the point:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)

# use retriever_chain to be able to use the chat history
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
chat_history = []

while True:
    question = input("> ")
    if question == "quit":
        break

    result = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })

    answer = result['answer']
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    print(answer)
