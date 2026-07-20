# Rasterização de PDF -> PNG (fundo de página, upload de PDF) — só
# serve a prévia do editor Konva, que não sabe desenhar PDF (o arquivo
# original é preservado à parte e mesclado como camada vetorial de
# verdade no PDF final, ver pdf_logic.py). PyMuPDF (fitz) escolhido por
# ser pip-instalável sozinho, sem depender de um binário de sistema
# (poppler) como pdf2image exigiria — mais simples de manter no Docker
# do Render.
import fitz  # PyMuPDF


def render_pdf_page_to_png(pdf_bytes: bytes, width_px: int, height_px: int) -> bytes:
    """Renderiza a PRIMEIRA página do PDF pra um PNG exatamente
    width_px x height_px — mesmo "fit: stretch" já usado pra fundo de
    página em imagem, então o zoom pode ser diferente em X e Y sem
    problema (preenche o quadro inteiro, sem manter proporção)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[0]
        page_w_pt, page_h_pt = page.rect.width, page.rect.height
        if page_w_pt <= 0 or page_h_pt <= 0:
            raise ValueError("Página do PDF com tamanho inválido.")

        zoom_x = width_px / page_w_pt
        zoom_y = height_px / page_h_pt
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()
