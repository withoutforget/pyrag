
from dataclasses import dataclass

import openai

LLMClient = openai.Client\

@dataclass(slots=True)
class LLMRequest:
    client: LLMClient
    model: str
    prompt: str

    async def ask(self, text: str, rag: list[dict]) -> str:
        input = {
            "request": text,
            "additional_data": rag,
        }

        response = self.client.responses.create(
            model = self.model,
            input = input,
            instructions = self.prompt,
        )

        if response.status != "completed":
            # TODO: implement logs + exceptions

            raise RuntimeError("Unknown status")
        
        return response.output_text
