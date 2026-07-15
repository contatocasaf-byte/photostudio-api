"""
Portado de removedor_fundo.py (Studio de Produtos, app desktop) —
remove_internal_background e rembg_process, linhas 177-339 do original.
`rembg_process` recebe bytes em vez de um path de arquivo local, já que
aqui a imagem vem do R2, não do disco.

As três funções de pós-processamento (remove_internal_background e
fill_enclosed_background_holes) são vetorizadas com numpy/scipy.ndimage
em vez de BFS em Python puro — mesma lógica (flood fill a partir da
borda == componente conexo que toca a borda), só que rodando em código
compilado. Numa foto de 12MP isso caiu de ~20s pra menos de 1s.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

# Conectividade-4 (só vizinhos ortogonais, sem diagonal) — mesma definição
# de "vizinho" usada no BFS original.
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _border_avg_color(rgb: np.ndarray) -> tuple[int, int, int]:
    """Cor média dos pixels da borda da imagem (todo o perímetro)."""
    border = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]]
    )
    r, g, b = (border.sum(axis=0) // len(border)).tolist()
    return r, g, b


def _labels_touching_border(labels: np.ndarray) -> np.ndarray:
    """Array booleano (mesmo shape) marcando pixels cujo componente conexo toca a borda."""
    border_ids = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_ids = border_ids[border_ids != 0]
    return np.isin(labels, border_ids)


def remove_internal_background(img_rgba: Image.Image,
                                img_original: Image.Image = None,
                                bg_tolerance: int = 40) -> Image.Image:
    """
    Pós-processamento após rembg para remover ilhas internas de fundo.

    O rembg frequentemente deixa pixels de fundo interno (ex: furo branco
    de uma peça) com alfa ALTO (próximo de 255), porque a rede os classifica
    como parte do objeto. A abordagem de verificar só alfa < 200 falha nesses
    casos.

    PASSO A — remoção por alfa: componente conexo de pixels com alfa < 128
    que NÃO toca a borda da imagem = ilha interna → alfa = 0. (Equivalente a
    "BFS pelas bordas": tudo que sobra fora do componente ligado à borda.)

    PASSO B — remoção por cor RGB (pega o que o rembg errou): componente
    conexo, na imagem ORIGINAL, de pixels com cor parecida com o fundo
    (média da borda) que TOCA a borda = fundo que o rembg deixou passar →
    alfa = 0.
    """
    arr = np.array(img_rgba)  # H, W, 4 — cópia própria (writable)
    alpha = arr[:, :, 3]

    # PASSO A
    low_alpha = alpha < 128
    labels_a, _ = ndimage.label(low_alpha, structure=_STRUCT4)
    connected_to_border_a = _labels_touching_border(labels_a)
    island_a = low_alpha & ~connected_to_border_a
    alpha[island_a] = 0

    if img_original is None:
        return Image.fromarray(arr, "RGBA")

    # PASSO B
    orig = img_original.convert("RGBA")
    if orig.size != (arr.shape[1], arr.shape[0]):
        orig = orig.resize((arr.shape[1], arr.shape[0]), Image.LANCZOS)
    orgb = np.array(orig)[:, :, :3].astype(np.int32)

    br, bg, bb = _border_avg_color(orgb)
    tol2 = bg_tolerance * bg_tolerance * 3
    dist2 = (orgb[:, :, 0] - br) ** 2 + (orgb[:, :, 1] - bg) ** 2 + (orgb[:, :, 2] - bb) ** 2
    bg_color_mask = dist2 <= tol2

    labels_b, _ = ndimage.label(bg_color_mask, structure=_STRUCT4)
    connected_to_border_b = _labels_touching_border(labels_b)
    alpha[connected_to_border_b] = 0

    return Image.fromarray(arr, "RGBA")


def fill_enclosed_background_holes(img_rgba: Image.Image, img_original: Image.Image,
                                    bg_tolerance: int = 40,
                                    max_hole_area_ratio: float = 0.15) -> Image.Image:
    """
    PASSO C — fecha furos internos totalmente cercados pelo objeto (ex:
    centro de uma arruela/mancal/bucha), que o Passo B não alcança.

    O Passo B só remove fundo que tem um caminho de pixels de cor
    parecida ligando até a borda da imagem. Um furo com o objeto formando
    um anel fechado ao redor não tem esse caminho — fica bloqueado pelo
    próprio objeto por todos os lados — e por isso o rembg deixa esses
    pixels com alfa alto (achando que são objeto) mesmo quando a cor
    deles é claramente de fundo.

    Busca componentes conexos de "cor de fundo + ainda opaco" em QUALQUER
    lugar da imagem, sem exigir ligação com a borda. Pra não arriscar
    apagar uma parte clara legítima do produto, só mexe em componentes
    que não tocam a borda e têm área pequena em relação ao total da
    imagem (`max_hole_area_ratio`).

    Nota: uma tentativa de tolerar furos "na sombra" (comparando cor sem
    brilho) foi testada e descartada — em fotos com objeto escuro fosco
    sob luz direcional, a cor do próprio objeto na sombra fica
    praticamente idêntica à cor do fundo na sombra (mesma iluminação),
    então nenhuma tolerância separa os dois casos de forma confiável.
    Furo em sombra forte continua sendo um caso pro editor manual (Fase
    2), não pro passo automático.
    """
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    orig = img_original.convert("RGBA")
    if orig.size != (w, h):
        orig = orig.resize((w, h), Image.LANCZOS)
    orgb = np.array(orig)[:, :, :3].astype(np.int32)

    br, bg, bb = _border_avg_color(orgb)
    tol2 = bg_tolerance * bg_tolerance * 3
    dist2 = (orgb[:, :, 0] - br) ** 2 + (orgb[:, :, 1] - bg) ** 2 + (orgb[:, :, 2] - bb) ** 2
    bg_color_mask = dist2 <= tol2

    candidate = bg_color_mask & (alpha >= 128)
    labels, num = ndimage.label(candidate, structure=_STRUCT4)
    if num == 0:
        return Image.fromarray(arr, "RGBA")

    sizes = ndimage.sum_labels(candidate, labels, index=np.arange(1, num + 1))
    border_ids = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    max_area = w * h * max_hole_area_ratio

    keep = np.zeros(num + 1, dtype=bool)  # índice 0 = fundo, ignorado
    for label_id, size in enumerate(sizes, start=1):
        if size <= max_area and label_id not in border_ids:
            keep[label_id] = True

    if keep.any():
        alpha[keep[labels]] = 0

    return Image.fromarray(arr, "RGBA")


def rembg_process(data: bytes, session, remove_holes: bool = True) -> Image.Image:
    """Retorna a imagem RGBA sem fundo (fluxo: rembg -> remove_internal_background -> fill_enclosed_background_holes).

    Fotos de celular costumam vir com orientação EXIF (o arquivo fica
    "deitado" no buffer de pixels cru, e só é mostrado em pé porque o
    visualizador aplica a rotação do metadado). Sem corrigir isso aqui, o
    pipeline processaria — e o app entregaria pro usuário — a imagem de
    lado. `exif_transpose` rotaciona fisicamente os pixels e já é chamado
    antes de qualquer outra coisa, então tudo depois (rembg, Passo A/B/C)
    trabalha na orientação correta.
    """
    from rembg import remove  # import pesado, adiado — ver get_session() em main.py

    original = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGBA")
    removed = remove(original, session=session).convert("RGBA")

    if remove_holes:
        removed = remove_internal_background(removed, img_original=original)
        removed = fill_enclosed_background_holes(removed, img_original=original)

    return removed
