import io
from dataclasses import dataclass

from PyPDF2 import PdfReader


@dataclass(slots=True)
class PDFParser:
    async def parse(self, data: bytes) -> str:
        pdf = PdfReader(io.BytesIO(data))
        return "\n".join(t.extract_text() for t in pdf.pages)
