from dishka import Provider, Scope, provide_all

from pyrag.infra.parsers.pdf import PDFParser
from pyrag.infra.parsers.html import HTMLParser
from pyrag.infra.parsers.splitter import TextSplitter
from pyrag.infra.parsers.telegram_json import TelegramJSONParser


class ParserProdiver(Provider):
    parsers = provide_all(
        PDFParser,
        HTMLParser,
        TextSplitter,
        TelegramJSONParser,
        scope=Scope.APP,
    )
