import ast
from abc import ABC
from typing import Dict, List

class ChatResponse(ABC):
    def __init__(self, content: str, total_tokens: int) -> None:
        self.content = content
        self.total_tokens = total_tokens
    
    def __repr__(self) -> str:
        return f"ChatResponse(content={self.content}, total_tokens={self.total_tokens})"

class BaseLLM(ABC):
    def __init__(self):
        pass

    def chat(self, messages: List[Dict]) -> ChatResponse:
        pass

    @staticmethod
    def literal_eval(response_content: str):
        response_content = response_content.strip()
        if response_content.startswith("```") and response_content.endswith("```"):
            if response_content.startswith("```python"):
                response_content = response_content[9:-3]
            elif response_content.startswith("```json"):
                response_content = response_content[7:-3]
            elif response_content.startswith("```str"):
                response_content = response_content[6:-3]
            elif response_content.startswith("```\n"):
                response_content = response_content[4:-3]
            else:
                response_content = response_content[3:-3]
        result = ast.literal_eval(response_content)
        return result