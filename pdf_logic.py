# Geração do PDF final do catálogo (Fase 5, Parte 10, Criador de
# Catálogos) — o payload já chega com toda a geometria/decisão de
# negócio resolvida pelo navegador (posições, merge de formas em linhas
# encostadas, versão de card-molde por item); este módulo só sabe
# DESENHAR. As únicas duas coisas reimplementadas aqui (o resto é
# geometria trivial) são as portas de:
#   - src/lib/canvasText.ts (wrapTextToLines/drawTextFit) -> busca de
#     tamanho de fonte + quebra de linha, usando pdfmetrics.stringWidth
#     no lugar de ctx.measureText.
#   - src/lib/fitImageOnCanvas.ts (getOpaqueBBox/getContentCenter/
#     fitImageOnCanvas) -> encaixe "contain" centralizado no conteúdo
#     opaco, usando Pillow+numpy no lugar de getImageData. O catálogo
#     nunca passa rotação/zoom/offset manual de foto (confirmado lendo
#     PreviewPageCanvas.tsx), então a porta não precisa suportar isso.
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from pdf_fonts import resolve_font
from pdf_models import GenerateCatalogPdfRequest, PdfBorderSpec, PdfImageSpec, PdfPdfBackgroundSpec, PdfShapeSpec, PdfTextSpec

# --- Coordenadas ------------------------------------------------------
# Espaço da app: px a 150dpi (mesma convenção de PX_PER_MM em
# pageConfig.ts), origem no canto superior esquerdo, Y crescendo pra
# baixo (igual ao Konva/Canvas 2D). reportlab: pontos (1/72"), origem
# inferior esquerda, Y crescendo pra cima. Cada coordenada é convertida
# explicitamente no ponto de uso (mais fácil de acertar do que compor
# via canvas.scale()/translate()).
PT_PER_PX = 72 / 150

# Resolução de fonte real (Google Font curada / fonte própria enviada)
# fica em pdf_fonts.py — "Arial" (padrão de todo card/página salvo
# antes da biblioteca de fontes existir, ou família não encontrada)
# cai em Helvetica, builtin do reportlab. Risco já avisado ao usuário:
# métricas ligeiramente diferentes podem divergir sutilmente do
# preview em casos extremos, mesmo com a fonte real embutida.


def px_to_pt(v_px: float) -> float:
    return v_px * PT_PER_PX


def px_point_to_pdf(x_px: float, y_px: float, pagina_altura_px: float) -> tuple[float, float]:
    """(x,y) medidos a partir do canto superior esquerdo -> ponto em
    espaço reportlab (origem inferior esquerda)."""
    return px_to_pt(x_px), px_to_pt(pagina_altura_px - y_px)


def box_to_pdf_rect(x_px: float, y_px: float, w_px: float, h_px: float, pagina_altura_px: float) -> tuple[float, float, float, float]:
    """(x,y) = canto superior esquerdo da caixa em espaço da app ->
    (x, y_do_canto_inferior, largura, altura) em pontos."""
    pdf_x = px_to_pt(x_px)
    pdf_y = px_to_pt(pagina_altura_px - (y_px + h_px))
    return pdf_x, pdf_y, px_to_pt(w_px), px_to_pt(h_px)


# --- Encaixe de foto (porta de fitImageOnCanvas.ts) --------------------

FILL_RATIO = 0.875  # TARGET_FILL_RATIO, mesmo valor do JS


def _bbox_at_threshold(alpha: np.ndarray, threshold: int) -> tuple[int, int, int, int] | None:
    mask = alpha > threshold
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def get_opaque_bbox(alpha: np.ndarray, threshold: int = 120) -> tuple[int, int, int, int] | None:
    """Porta de getOpaqueBBox (fitImageOnCanvas.ts): threshold alto por
    padrão pra ignorar sombra suave semi-transparente; sem pixel sólido
    nesse threshold, cai pra um threshold bem mais baixo em vez de
    tratar a imagem inteira como vazia."""
    bbox = _bbox_at_threshold(alpha, threshold)
    if bbox:
        return bbox
    return _bbox_at_threshold(alpha, 10)


def get_content_center(alpha: np.ndarray, bbox: tuple[int, int, int, int], threshold: int = 120) -> tuple[float, float]:
    """Porta de getContentCenter: centro de MASSA do conteúdo opaco
    dentro da bbox (não o centro geométrico do retângulo)."""
    left, top, right, bottom = bbox
    fallback_cx = (left + right) / 2
    fallback_cy = (top + bottom) / 2
    rw = right - left
    rh = bottom - top
    if rw <= 0 or rh <= 0:
        return fallback_cx, fallback_cy

    region = alpha[top:bottom, left:right] > threshold
    col_sum = region.sum(axis=0).astype(np.float64)
    row_sum = region.sum(axis=1).astype(np.float64)

    sum_w = col_sum.sum()
    if sum_w <= 0:
        return fallback_cx, fallback_cy

    xs = np.arange(rw)
    sum_wx = float((col_sum * xs).sum())
    sum_w_rows = float(row_sum.sum())
    ys = np.arange(rh)
    sum_wy = float((row_sum * ys).sum())

    cx_local = sum_wx / sum_w
    cy_local = (sum_wy / sum_w_rows) if sum_w_rows > 0 else rh / 2
    return left + cx_local, top + cy_local


def _resize_rgba_premultiplied(img: Image.Image, size: tuple[int, int], resample) -> Image.Image:
    """Redimensiona uma imagem RGBA sem o halo escuro que aparece perto
    da borda de objetos claros contra fundo branco — o Pillow interpola
    R/G/B/A de forma independente (sem pré-multiplicar), então pixels
    quase-transparentes perto da borda do objeto "vazam" sua cor RGB
    crua (tipicamente preta, sobrando do fundo removido) pro resultado
    reamostrado. Pré-multiplicar antes de reamostrar e desfazer depois
    é o fix padrão desse problema conhecido do Pillow. Invisível em
    objetos escuros (o halo se confunde com o próprio objeto) — só
    ficou visível numa peça metálica clara contra a página branca."""
    r, g, b, a = img.split()
    a_arr = np.asarray(a, dtype=np.float32) / 255.0
    rgb_arr = np.asarray(Image.merge("RGB", (r, g, b)), dtype=np.float32)
    premultiplied = Image.fromarray((rgb_arr * a_arr[..., None]).astype(np.uint8), "RGB")

    premultiplied_resized = premultiplied.resize(size, resample)
    a_resized = a.resize(size, resample)

    a_resized_arr = np.asarray(a_resized, dtype=np.float32) / 255.0
    premultiplied_resized_arr = np.asarray(premultiplied_resized, dtype=np.float32)
    safe_alpha = np.maximum(a_resized_arr, 1e-6)
    rgb_resized = np.clip(premultiplied_resized_arr / safe_alpha[..., None], 0, 255).astype(np.uint8)

    return Image.merge("RGBA", (*Image.fromarray(rgb_resized, "RGB").split(), a_resized))


def fit_image_on_box(img: Image.Image, box_w: float, box_h: float) -> Image.Image:
    """Porta de fitImageOnCanvas (transform sempre {} — catálogo nunca
    usa rotação/zoom/offset manual): região de interesse simétrica em
    torno do centro de massa, encaixada por inteiro na caixa (sem
    cortar) a ~87.5% de preenchimento. Devolve uma imagem RGBA já do
    tamanho exato box_w x box_h, pronta pra desenhar sem mais cálculo."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    alpha = np.asarray(rgba)[:, :, 3]

    bbox = get_opaque_bbox(alpha, 120)
    if bbox is None:
        roi = rgba
        rw, rh = w, h
    else:
        cx, cy = get_content_center(alpha, bbox, 120)
        left, top, right, bottom = bbox
        half_w = max(cx - left, right - cx)
        half_h = max(cy - top, bottom - cy)
        sym_left = max(0.0, cx - half_w)
        sym_top = max(0.0, cy - half_h)
        sym_right = min(float(w), cx + half_w)
        sym_bottom = min(float(h), cy + half_h)

        r_left = round(sym_left)
        r_top = round(sym_top)
        rw = max(1, round(sym_right) - r_left)
        rh = max(1, round(sym_bottom) - r_top)
        roi = rgba.crop((r_left, r_top, r_left + rw, r_top + rh))

    if rw <= 0 or rh <= 0:
        return Image.new("RGBA", (max(1, round(box_w)), max(1, round(box_h))), (0, 0, 0, 0))

    scale = min(box_w / rw, box_h / rh) * FILL_RATIO
    nw = max(1, round(rw * scale))
    nh = max(1, round(rh * scale))
    resized = _resize_rgba_premultiplied(roi, (nw, nh), Image.LANCZOS)

    dest_x = round((box_w - nw) / 2)
    dest_y = round((box_h - nh) / 2)

    out = Image.new("RGBA", (max(1, round(box_w)), max(1, round(box_h))), (0, 0, 0, 0))
    out.paste(resized, (dest_x, dest_y), resized)
    return out


# --- Texto (porta de canvasText.ts) ------------------------------------


def wrap_text_to_lines(
    text: str, font_name: str, size: float, max_w: float, max_lines: int, allow_truncate: bool
) -> tuple[list[str], bool]:
    """Porta de wrapTextToLines: quebra gulosa por palavra, usando
    pdfmetrics.stringWidth no lugar de ctx.measureText."""
    words = text.split()
    if not words:
        return [text], False

    lines: list[str] = []
    current = ""
    broke_early = False

    for word in words:
        trial = (current + " " + word).strip()
        fits = pdfmetrics.stringWidth(trial, font_name, size) <= max_w
        if fits or not current:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                broke_early = True
                break
    if not broke_early and current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    used_words = sum(len(ln.split()) for ln in lines)
    overflow = used_words < len(words)

    if overflow and allow_truncate and lines:
        last = lines[-1]
        while True:
            fits = pdfmetrics.stringWidth(last + "…", font_name, size) <= max_w
            if fits or len(last) <= 1:
                break
            last = last[:-1].rstrip()
        lines[-1] = last + "…"

    return (lines if lines else [text]), overflow


def _draw_line(c: "canvas.Canvas", text: str, x_px: float, baseline_y_px: float, pagina_altura_px: float) -> None:
    x_pt, y_pt = px_point_to_pdf(x_px, baseline_y_px, pagina_altura_px)
    c.drawString(x_pt, y_pt, text)


# Proporção descent/capHeight do Helvetica (0.207/0.718) — usada como
# aproximação universal do "quanto um 'g' desce" quando a fonte não
# expõe essa métrica diretamente (ver _tight_ascent_descent).
_DESCENT_RATIO = 0.207 / 0.718


def _tight_ascent_descent(font_name: str, size: float) -> tuple[float, float]:
    """Ascent/descent "justos", pra bater com o que o navegador mede via
    ctx.measureText("Ag").actualBoundingBoxAscent/Descent (bounding box
    real dos glifos desenhados) — usado no editor/preview (canvasText.ts).

    pdfmetrics.getAscentDescent() devolve a métrica "hhea" da fonte
    inteira, pensada pra reservar espaço pra QUALQUER glifo dela
    (acentos, swashes, etc.), não o texto real desenhado. Pra fontes
    decorativas embutidas (Google Fonts como Bebas Neue/Abril Fatface)
    isso vem bem maior que o necessário — confirmado: Abril Fatface tem
    ascent hhea de 105.8% do tamanho da fonte, contra ~71.8% do
    Helvetica — e como baseline_y = y + ascent, esse excesso empurra a
    linha de base (e o texto inteiro) pra baixo dentro do card,
    divergindo visivelmente do editor/preview (texto encostando na
    base da forma em vez de centralizado verticalmente, print real do
    usuário comparando os dois).

    capHeight (altura de "A" maiúsculo) é uma métrica bem mais estável
    entre fontes (~70-84% do tamanho, testado em 6 fontes reais da
    lista curada) e corresponde de perto ao que o navegador mede pra
    "A" — usa ela como ascent "justo" quando disponível (fontes TTF
    embutidas via TTFont); fontes builtin do reportlab (Helvetica) não
    expõem capHeight e caem no getAscentDescent original, que já é
    coerente o bastante nelas."""
    try:
        cap_height = pdfmetrics.getFont(font_name).face.capHeight
    except Exception:
        cap_height = 0
    if not cap_height or cap_height <= 0:
        return pdfmetrics.getAscentDescent(font_name, size)
    ascent = cap_height / 1000 * size
    return ascent, -ascent * _DESCENT_RATIO


def draw_text_fit(c: "canvas.Canvas", spec: PdfTextSpec, pagina_altura_px: float, font_name: str) -> None:
    """Porta de drawTextFit: busca linear do maior tamanho de fonte (até
    o piso) que quebra em <= maxLines linhas sem estourar maxW; se nem
    no piso couber, cai pro fallback de reticências. `font_name` já vem
    resolvido (pdf_fonts.resolve_font) — real ou Helvetica de fallback."""
    floor = min(spec.minSize, spec.fontSizeMax)
    lines = [spec.text]
    used_size = floor
    found = False

    size = spec.fontSizeMax
    while size >= floor:
        result_lines, overflow = wrap_text_to_lines(spec.text, font_name, size, spec.maxW, spec.maxLines, False)
        fits = all(pdfmetrics.stringWidth(ln, font_name, size) <= spec.maxW for ln in result_lines)
        if fits and not overflow:
            lines = result_lines
            used_size = size
            found = True
            break
        size -= 1

    if not found:
        used_size = floor
        lines, _ = wrap_text_to_lines(spec.text, font_name, floor, spec.maxW, spec.maxLines, True)

    ascent, descent = _tight_ascent_descent(font_name, used_size)
    line_h = (ascent + abs(descent)) * spec.lineSpacing

    c.setFont(font_name, px_to_pt(used_size))
    c.setFillColor(HexColor(spec.color))
    c.setFillAlpha(1.0)

    for i, line in enumerate(lines):
        line_w = pdfmetrics.stringWidth(line, font_name, used_size)
        baseline_y_px = spec.y + i * line_h + ascent

        if spec.align == "center":
            _draw_line(c, line, spec.x + (spec.maxW - line_w) / 2, baseline_y_px, pagina_altura_px)
        elif spec.align == "right":
            _draw_line(c, line, spec.x + (spec.maxW - line_w), baseline_y_px, pagina_altura_px)
        elif spec.align == "justify" and i < len(lines) - 1 and " " in line.strip():
            words = line.split(" ")
            words_w = sum(pdfmetrics.stringWidth(w, font_name, used_size) for w in words)
            gap_count = max(1, len(words) - 1)
            space_w = pdfmetrics.stringWidth(" ", font_name, used_size)
            extra_space = max(0.0, spec.maxW - words_w) / gap_count
            cursor_x = spec.x
            for w in words:
                _draw_line(c, w, cursor_x, baseline_y_px, pagina_altura_px)
                cursor_x += pdfmetrics.stringWidth(w, font_name, used_size) + space_w + extra_space
        else:
            _draw_line(c, line, spec.x, baseline_y_px, pagina_altura_px)


# --- Formas / bordas / imagens (geometria trivial, já resolvida pelo payload) ---


def draw_shape(c: "canvas.Canvas", spec: PdfShapeSpec, pagina_altura_px: float) -> None:
    c.setFillColor(HexColor(spec.color))
    c.setFillAlpha(spec.opacity)

    if spec.shapeType == "retangulo":
        x, y, w, h = box_to_pdf_rect(spec.x, spec.y, spec.w, spec.h, pagina_altura_px)
        c.rect(x, y, w, h, fill=1, stroke=0)
    elif spec.shapeType == "elipse":
        x, y, w, h = box_to_pdf_rect(spec.x, spec.y, spec.w, spec.h, pagina_altura_px)
        c.ellipse(x, y, x + w, y + h, fill=1, stroke=0)
    else:  # triangulo — mesmos vértices do sceneFunc do Konva
        # (moveTo(w/2,0) lineTo(w,h) lineTo(0,h)): topo-centro,
        # base-direita, base-esquerda, em espaço da app.
        p1 = px_point_to_pdf(spec.x + spec.w / 2, spec.y, pagina_altura_px)
        p2 = px_point_to_pdf(spec.x + spec.w, spec.y + spec.h, pagina_altura_px)
        p3 = px_point_to_pdf(spec.x, spec.y + spec.h, pagina_altura_px)
        path = c.beginPath()
        path.moveTo(*p1)
        path.lineTo(*p2)
        path.lineTo(*p3)
        path.close()
        c.drawPath(path, fill=1, stroke=0)


def draw_border(c: "canvas.Canvas", spec: PdfBorderSpec, pagina_altura_px: float) -> None:
    x1, y1 = px_point_to_pdf(spec.x1, spec.y1, pagina_altura_px)
    x2, y2 = px_point_to_pdf(spec.x2, spec.y2, pagina_altura_px)
    c.setStrokeColor(HexColor(spec.color))
    c.setStrokeAlpha(spec.opacity)
    c.setLineWidth(px_to_pt(spec.width))
    c.line(x1, y1, x2, y2)


def draw_image(c: "canvas.Canvas", spec: PdfImageSpec, pagina_altura_px: float, fitted_image: Image.Image) -> None:
    # setFillAlpha é estado do CANVAS, não por elemento — sem resetar
    # aqui, uma forma decorativa desenhada antes (draw_shape) com
    # opacidade < 1 vaza pra próxima imagem desenhada, deixando-a
    # semitransparente até o próximo draw_text_fit resetar (linha 246).
    c.setFillAlpha(1.0)
    x, y, w, h = box_to_pdf_rect(spec.x, spec.y, spec.w, spec.h, pagina_altura_px)
    c.drawImage(ImageReader(fitted_image), x, y, width=w, height=h, mask="auto")


# --- Orquestrador -------------------------------------------------------


PREFETCH_WORKERS = 8


class ImageStore:
    """Imagens do R2 do catálogo inteiro, buscadas em paralelo ANTES de
    desenhar (antes era 1 requisição sequencial por imagem — catálogo
    grande passava de 1 minuto só nisso). Guarda os bytes crus (leves) e
    decodifica sob demanda: só fica decodificada na memória a imagem
    usada em mais de um lugar (logo/fundo repetidos em várias páginas) —
    foto de produto usada uma vez é descartada logo depois de desenhada,
    em vez de acumular centenas de RGBA em memória.

    Imagem que não pôde ser baixada/decodificada não derruba o PDF
    inteiro: `get` devolve None, o chamador simplesmente não desenha, e
    `failed` conta quantas ficaram de fora."""

    def __init__(self, client, bucket: str, usage: dict[str, int]):
        self._usage = usage
        self._decoded: dict[str, Image.Image] = {}
        self._raw: dict[str, bytes] = {}
        self.failed: set[str] = set()

        def fetch(key: str) -> tuple[str, bytes | None]:
            try:
                return key, client.get_object(Bucket=bucket, Key=key)["Body"].read()
            except Exception:
                return key, None

        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
            for key, data in pool.map(fetch, usage.keys()):
                if data is None:
                    self.failed.add(key)
                else:
                    self._raw[key] = data

    def get(self, key: str) -> Image.Image | None:
        if key in self._decoded:
            return self._decoded[key]
        data = self._raw.get(key)
        if data is None:
            return None
        try:
            img = Image.open(BytesIO(data)).convert("RGBA")
        except Exception:
            self.failed.add(key)
            return None
        if self._usage.get(key, 0) > 1:
            self._decoded[key] = img
        return img


def _collect_image_usage(payload: GenerateCatalogPdfRequest) -> dict[str, int]:
    """Quantas vezes cada chave de imagem aparece no payload (inclui
    fundos raster; fundo em PDF é tratado à parte)."""
    usage: dict[str, int] = {}

    def add(key: str) -> None:
        usage[key] = usage.get(key, 0) + 1

    for page in payload.pages:
        if isinstance(page.background, PdfImageSpec):
            add(page.background.key)
        for op in (*page.pageFields, *page.cardFields):
            if isinstance(op, PdfImageSpec):
                add(op.key)
    return usage


def _draw_field_op(
    c: "canvas.Canvas", op, pagina_altura_px: float, store: ImageStore, client, bucket: str
) -> None:
    if isinstance(op, PdfImageSpec):
        img = store.get(op.key)
        if img is None:
            return
        if op.fit == "stretch":
            fitted = img.resize((max(1, round(op.w)), max(1, round(op.h))))
        else:
            fitted = fit_image_on_box(img, op.w, op.h)
        draw_image(c, op, pagina_altura_px, fitted)
    else:
        font_name = resolve_font(op.fontFamily, op.fontWeight == "bold", client, bucket)
        draw_text_fit(c, op, pagina_altura_px, font_name)


def build_pdf(payload: GenerateCatalogPdfRequest, client, bucket: str) -> tuple[bytes, int]:
    """Monta o PDF página a página, respeitando a MESMA ordem de
    camadas do preview (PreviewPageCanvas.tsx): fundo -> formas de
    página -> campos de cabeçalho/rodapé -> bordas de card -> formas de
    card -> campos de card. Cada `page.showPage()` fecha a página atual
    e abre a seguinte.

    Fundo em PDF (upload de PDF, ver pdf_render.py) é um caso à parte:
    a página do PDF original NÃO é desenhada aqui (reportlab não sabe
    desenhar PDF dentro de PDF) — só fica marcada em `pages_with_pdf_bg`
    e é mesclada como camada vetorial de base num segundo passo, depois
    de fechado o PDF de conteúdo inteiro (ver _merge_pdf_backgrounds)."""
    buf = BytesIO()
    page_w_pt = px_to_pt(payload.paginaLargura)
    page_h_pt = px_to_pt(payload.paginaAltura)
    c = canvas.Canvas(buf, pagesize=(page_w_pt, page_h_pt))

    store = ImageStore(client, bucket, _collect_image_usage(payload))
    pdf_bg_cache: dict[str, bytes] = {}
    pages_with_pdf_bg: dict[int, str] = {}

    for i, page in enumerate(payload.pages):
        if page.background:
            if isinstance(page.background, PdfPdfBackgroundSpec):
                pages_with_pdf_bg[i] = page.background.key
                if page.background.key not in pdf_bg_cache:
                    obj = client.get_object(Bucket=bucket, Key=page.background.key)
                    pdf_bg_cache[page.background.key] = obj["Body"].read()
            else:
                img = store.get(page.background.key)
                if img is not None:
                    fitted = img.resize((max(1, round(page.background.w)), max(1, round(page.background.h))))
                    draw_image(c, page.background, payload.paginaAltura, fitted)

        for shape in page.pageShapes:
            draw_shape(c, shape, payload.paginaAltura)

        for op in page.pageFields:
            _draw_field_op(c, op, payload.paginaAltura, store, client, bucket)

        for border in page.cardBorders:
            draw_border(c, border, payload.paginaAltura)

        for shape in page.cardShapes:
            draw_shape(c, shape, payload.paginaAltura)

        for op in page.cardFields:
            _draw_field_op(c, op, payload.paginaAltura, store, client, bucket)

        c.showPage()

    c.save()
    content_bytes = buf.getvalue()

    skipped = len(store.failed)
    if not pages_with_pdf_bg:
        return content_bytes, skipped
    return _merge_pdf_backgrounds(content_bytes, pages_with_pdf_bg, pdf_bg_cache, page_w_pt, page_h_pt), skipped


def _merge_pdf_backgrounds(
    content_bytes: bytes,
    pages_with_pdf_bg: dict[int, str],
    pdf_bg_cache: dict[str, bytes],
    page_w_pt: float,
    page_h_pt: float,
) -> bytes:
    """Segundo passo, só quando ao menos uma página usa fundo em PDF:
    pra cada página marcada, pega a 1ª página do PDF de fundo original
    (escalada pro tamanho exato da página do catálogo — mesma semântica
    "stretch" já usada pro fundo em imagem), mescla o conteúdo gerado
    por cima dela (`merge_page` desenha o argumento SOBRE quem chama) e
    usa isso como a página final. Reabre um PdfReader novo por uso (não
    reaproveita a mesma PageObject entre páginas) porque `merge_page`
    muta o objeto em memória — reaproveitar acumularia conteúdo de
    página errada."""
    content_reader = PdfReader(BytesIO(content_bytes))
    writer = PdfWriter()

    for i, content_page in enumerate(content_reader.pages):
        bg_key = pages_with_pdf_bg.get(i)
        if bg_key is None:
            writer.add_page(content_page)
            continue

        bg_reader = PdfReader(BytesIO(pdf_bg_cache[bg_key]))
        bg_page = bg_reader.pages[0]
        bg_page.scale_to(page_w_pt, page_h_pt)
        bg_page.merge_page(content_page)
        writer.add_page(bg_page)

    out = BytesIO()
    writer.write(out)
    return out.getvalue()
