from dishka import Provider, Scope, provide_all

from pyrag.infra.parsers.pdf import PDFParser
from pyrag.infra.parsers.html import HTMLParser
from pyrag.infra.parsers.splitter import TextSplitter


class ParserProdiver(Provider):
    parsers = provide_all(
        PDFParser,
        HTMLParser,
        TextSplitter,
        scope=Scope.APP,
    )
