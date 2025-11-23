
from dataclasses import dataclass
import json
import logging

import openai

LLMClient = openai.Client\

@dataclass(slots=True)
class LLMRequest:
    client: LLMClient
    model: str
    prompt: str

    async def ask(self, text: str, rag: list[dict]) -> str:
        input = "{rag}\n\n{text}".format(
            rag = rag, text = text
        )

        logging.critical("input: `%s`", input)

        response = self.client.responses.create(
            model = self.model,
            input = input,
            instructions = self.prompt,
        )

        if response.status != "completed":
            # TODO: implement logs + exceptions

            raise RuntimeError("Unknown status")
        
        return response.output_text
