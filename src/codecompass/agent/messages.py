class Message(dict):
    """Base message that acts like a dict"""
    
    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)
        self.role = role
        self.content = content


class HumanMessage(Message):
    def __init__(self, content: str):
        super().__init__("user", content)


class AIMessage(Message):
    def __init__(self, content: str):
        super().__init__("assistant", content)


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__("system", content)