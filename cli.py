from langchain_core.messages import HumanMessage, AIMessage
from chain import create_handbook_retrieval_chain
from embedding import load_embeddings

vector = load_embeddings()
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
