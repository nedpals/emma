import asyncio
import logging
import os
import sys

from agent_setup import create_agent
from models import UserMessage, AIMessage

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="  %(name)s: %(message)s")

agent = create_agent()
chat_history: list[dict] = []


async def main():
    while True:
        question = input("> ")
        if question == "quit":
            break

        answer = ""
        async for event in agent.run(question, chat_history):
            if event["type"] == "tool_start":
                print(f"  [Using {event['tool']}...]")
            elif event["type"] == "answer_chunk":
                sys.stdout.write(event["chunk"])
                sys.stdout.flush()
                answer += event["chunk"]
            elif event["type"] == "answer_done":
                print()  # newline after streaming
            elif event["type"] == "error":
                print(f"  Error: {event['message']}")

        if answer:
            chat_history.append(UserMessage(question))
            chat_history.append(AIMessage(answer))


if __name__ == "__main__":
    asyncio.run(main())
