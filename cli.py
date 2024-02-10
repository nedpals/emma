from langchain_core.messages import HumanMessage, AIMessage
from chain import create_handbook_retrieval_chain
from embedding import load_vector_store

retrieval_chain = create_handbook_retrieval_chain(load_vector_store())
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
