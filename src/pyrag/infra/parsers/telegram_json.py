from dataclasses import dataclass
import json


@dataclass(slots=True)
class TelegramJSONParser:
    async def parse(self, data: bytes) -> str:
        payload = json.loads(data)
        messages = payload if isinstance(payload, list) else payload.get("messages", [])
        return "\n".join(self._format(m) for m in messages if m.get("type") == "message")

    def _format(self, m: dict) -> str:
        date = m.get("date", "")[:10]
        sender = m.get("from") or m.get("actor") or "Unknown"
        text = self._extract_text(m.get("text", ""))
        return f"[{date}] {sender}: {text}"

    def _extract_text(self, text: str | list) -> str:
        if isinstance(text, str):
            return text
        # text может быть списком сегментов: [{"type": "bold", "text": "..."}, ...]
        return "".join(
            seg if isinstance(seg, str) else seg.get("text", "")
            for seg in text
        )