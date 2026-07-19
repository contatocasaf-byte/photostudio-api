# Modelos Pydantic do payload de exportação de PDF (Fase 5, Parte 10,
# Criador de Catálogos) — espelham 1:1 os tipos TypeScript definidos em
# catalogos/core/pdfExport.ts (frontend). O payload já vem com toda a
# geometria/decisão de negócio resolvida pelo navegador (posições,
# merge de formas em linhas encostadas, versão de card-molde por item)
# — este backend só desenha.
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

TextAlign = Literal["left", "center", "right", "justify"]
FontWeight = Literal["bold", "normal"]
ShapeType = Literal["retangulo", "elipse", "triangulo"]
ImageFit = Literal["contain", "stretch"]


class PdfImageSpec(BaseModel):
    kind: Literal["image"]
    key: str
    x: float
    y: float
    w: float
    h: float
    fit: ImageFit


class PdfTextSpec(BaseModel):
    kind: Literal["text"]
    text: str
    x: float
    y: float
    maxW: float
    fontSizeMax: float
    minSize: float = 8.0
    color: str
    align: TextAlign
    fontWeight: FontWeight
    maxLines: int
    lineSpacing: float = 1.15


# Discriminador explícito por "kind" — mais robusto que deixar o
# Pydantic inferir pela forma dos campos (image/text têm formas bem
# diferentes, mas discriminador explícito não corre risco nenhum de
# ambiguidade e falha rápido/claro se o payload vier malformado).
PdfFieldSpec = Annotated[Union[PdfImageSpec, PdfTextSpec], Field(discriminator="kind")]


class PdfShapeSpec(BaseModel):
    shapeType: ShapeType
    x: float
    y: float
    w: float
    h: float
    color: str
    opacity: float


class PdfBorderSpec(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    color: str
    opacity: float
    width: float


class PdfPageSpec(BaseModel):
    background: Optional[PdfImageSpec] = None
    pageFields: list[PdfFieldSpec]
    cardBorders: list[PdfBorderSpec]
    cardShapes: list[PdfShapeSpec]
    cardFields: list[PdfFieldSpec]


class GenerateCatalogPdfRequest(BaseModel):
    paginaLargura: float
    paginaAltura: float
    pages: list[PdfPageSpec]
