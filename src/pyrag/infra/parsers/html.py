import io
from dataclasses import dataclass
from lxml.html import fromstring

from PyPDF2 import PdfReader


@dataclass(slots=True)
class HTMLParser:
    async def parse(self, data: bytes) -> str:
        return fromstring(
            html = data.decode()
        ).text_content()