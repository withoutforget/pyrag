import logging
from dataclasses import dataclass

import openai

LLMClient = openai.Client

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMRequest:
    client: LLMClient
    model: str
    prompt: str

    async def ask(self, text: str, rag: list[dict]) -> str:
        inp = f"{rag}\n\n{text}"

        logger.critical("input: `%s`", inp)

        response = self.client.responses.create(
            model=self.model,
            input=inp,
            instructions=self.prompt,
        )

        if response.status != "completed":
            # TODO: implement logs + exceptions

            raise RuntimeError("Unknown status")  # noqa

        return response.output_text
