from llm import embeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores.chroma import Chroma
from chain import create_handbook_retrieval_chain

vector = Chroma(persist_directory="./embeddings_db", embedding_function=embeddings)
retrieval_chain = create_handbook_retrieval_chain(vector)
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
