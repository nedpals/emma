from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(role="system", content=content)

class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__(role="user", content=content)

class AIMessage(Message):
    def __init__(self, content: str):
        super().__init__(role="assistant", content=content)