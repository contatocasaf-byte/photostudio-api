"""
Teste manual, sem R2 nem servidor HTTP: valida que a lógica portada
(rembg_logic.py) roda de ponta a ponta contra uma imagem real do lote de
produtos Brasmam. Rodar com: .venv\\Scripts\\python.exe test_local.py <caminho_da_imagem>
"""
import sys

from rembg import new_session

from rembg_logic import rembg_process

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, "rb") as f:
        data = f.read()

    print("Carregando modelo u2net (primeira vez baixa ~170MB)...")
    session = new_session("u2net")

    print("Removendo fundo...")
    result = rembg_process(data, session)

    out_path = path.rsplit(".", 1)[0] + "_sem_fundo.png"
    result.save(out_path)
    print(f"Salvo em: {out_path}")
