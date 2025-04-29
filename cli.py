from models import UserMessage, AIMessage
from prompt import RetrievalChain
from vector_store import load_vector_store

retrieval_chain = RetrievalChain(load_vector_store(), history_aware=False)
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
    chat_history.append(UserMessage(question))
    chat_history.append(AIMessage(answer))

    print(answer)
