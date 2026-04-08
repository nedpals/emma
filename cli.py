import asyncio
import logging

from agent_setup import create_agent

logging.basicConfig(level=logging.INFO, format="  %(name)s: %(message)s")

agent = create_agent()
chat_history = []


async def main():
    while True:
        question = input("> ")
        if question == "quit":
            break

        answer = ""
        async for event in agent.run(question, chat_history):
            if event["type"] == "tool_start":
                print(f"  [Using {event['tool']}...]")
            elif event["type"] == "answer":
                answer = event["answer"]
            elif event["type"] == "error":
                print(f"  Error: {event['message']}")

        if answer:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
