from dishka import Provider, Scope
from dishka import provide, provide_all

from pyrag.infra.parsers.pdf import PDFParser
from pyrag.infra.parsers.splitter import TextSplitter

class ParserProdiver(Provider):
    parsers = provide_all(
        PDFParser,
        TextSplitter,
        scope = Scope.APP,
    )