# Resolução e registro de fontes reais no PDF final (Fase 5, Parte 11
# "voltaremos pro PDF depois") — porta da mesma ordem de resolução já
# usada no navegador (src/lib/fonts/fontLoader.ts): fonte PRÓPRIA
# enviada pelo usuário (R2, prefixo fontes/) primeiro, senão Google
# Font curada, senão cai pro padrão do sistema (Arial -> Helvetica,
# builtin do reportlab, sem precisar embutir nada).
#
# Cache em memória de PROCESSO (não por requisição/geração): o
# conteúdo de uma família+peso não muda durante a vida do deploy, só
# busca/registra uma vez por instância do backend.
import re
from io import BytesIO

import requests
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_PREFIX = "fontes/"
FONT_EXTS = ["ttf", "otf", "woff2", "woff"]

_registered: dict[tuple[str, bool], str] = {}
_google_ttf_cache: dict[tuple[str, int], bytes | None] = {}


def _safe_family(family: str) -> str:
    """Mesma transformação de src/lib/fonts/actions.ts (getFontUploadUrl)
    — precisa bater exatamente com a chave usada no upload."""
    cleaned = re.sub(r"[^A-Za-z0-9 ]", "", family.strip())
    return re.sub(r"\s+", "_", cleaned)


def _woff_to_ttf(data: bytes) -> bytes | None:
    try:
        from fontTools.ttLib import TTFont as FTFont

        font = FTFont(BytesIO(data))
        font.flavor = None
        out = BytesIO()
        font.save(out)
        return out.getvalue()
    except Exception:
        return None


def _fetch_custom_font(family: str, weight: int, client, bucket: str) -> bytes | None:
    safe = _safe_family(family)
    if not safe:
        return None
    for ext in FONT_EXTS:
        key = f"{FONT_PREFIX}{safe}-{weight}.{ext}"
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
        except Exception:
            continue
        if ext in ("woff", "woff2"):
            data = _woff_to_ttf(data)
            if data is None:
                continue
        return data
    return None


def _fetch_google_font_ttf(family: str, weight: int) -> bytes | None:
    """Baixa e converte pra TTF puro a variante família+peso de uma
    fonte curada. A API CSS2 do Google serve o formato mais moderno que
    o User-Agent da requisição aceitar (hoje, WOFF2) — em vez de tentar
    forçar TTF direto via User-Agent antigo (técnica que já se mostrou
    frágil: o Google pode mudar o que cada UA recebe a qualquer
    momento), aceita o que vier e reaproveita a mesma conversão via
    fonttools já usada pras fontes PRÓPRIAS em WOFF/WOFF2."""
    cache_key = (family, weight)
    if cache_key in _google_ttf_cache:
        return _google_ttf_cache[cache_key]

    result: bytes | None = None
    try:
        # URL montada crua (não via params=) — o "+" que separa palavras
        # do nome da família (exigido pela API CSS2 do Google) precisa
        # ficar literal na query string; requests.get(params=...) sempre
        # re-codifica valores, transformando "+" em "%2B" (que o Google
        # não entende como espaço), e a API devolve 400 pra QUALQUER
        # família com espaço no nome (Bebas Neue, Abril Fatface etc.) —
        # famílias de uma palavra só (Montserrat) mascaravam esse bug.
        family_param = family.replace(" ", "+")
        url = f"https://fonts.googleapis.com/css2?family={family_param}:wght@{weight}&display=swap"
        css_res = requests.get(url, timeout=10)
        if css_res.ok:
            match = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css_res.text)
            if match:
                font_url = match.group(1)
                font_res = requests.get(font_url, timeout=10)
                if font_res.ok:
                    data = font_res.content
                    result = data if font_url.endswith(".ttf") else _woff_to_ttf(data)
    except Exception:
        result = None

    _google_ttf_cache[cache_key] = result
    return result


def resolve_font(family: str | None, bold: bool, client, bucket: str) -> str:
    """Retorna o nome de fonte já registrado no reportlab, pronto pra
    c.setFont(). "Arial" (valor padrão de todo card/página, inclusive
    os salvos antes da biblioteca de fontes existir) e famílias vazias/
    não resolvidas caem direto em Helvetica, sem nenhuma rede/R2 —
    caminho mais comum do app inteiro continua com custo zero."""
    fallback = "Helvetica-Bold" if bold else "Helvetica"
    if not family or family == "Arial":
        return fallback

    cache_key = (family, bold)
    if cache_key in _registered:
        return _registered[cache_key]

    weight = 700 if bold else 400
    data = _fetch_custom_font(family, weight, client, bucket) or _fetch_google_font_ttf(family, weight)

    # Sem a variante pedida (ex.: só existe "regular" enviado pra essa
    # família) — tenta o outro peso antes de desistir da família
    # inteira, evita perder uma fonte real por causa só do negrito.
    if data is None:
        alt_weight = 400 if bold else 700
        data = _fetch_custom_font(family, alt_weight, client, bucket) or _fetch_google_font_ttf(family, alt_weight)

    if data is None:
        _registered[cache_key] = fallback
        return fallback

    internal_name = f"{family}-{'Bold' if bold else 'Regular'}"
    try:
        pdfmetrics.registerFont(TTFont(internal_name, BytesIO(data)))
        _registered[cache_key] = internal_name
        return internal_name
    except Exception:
        _registered[cache_key] = fallback
        return fallback
