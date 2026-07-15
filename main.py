import os
import uuid
from io import BytesIO
from typing import Optional

import boto3
from botocore.config import Config
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rembg_logic import rembg_process

app = FastAPI()

BUCKET = os.environ.get("R2_BUCKET_NAME", "")
SHARED_SECRET = os.environ.get("BACKEND_SHARED_SECRET", "")

_session = None


def get_session():
    # Import de `rembg` fica dentro da função de propósito: puxa
    # onnxruntime/numba/opencv/scikit-image, cuja importação sozinha já leva
    # minutos numa CPU fraca (Render free) — se isso acontecesse no nível do
    # módulo, o Uvicorn não conseguiria nem abrir a porta a tempo do Render
    # considerar o deploy no ar. Adiando pra cá, o servidor abre a porta
    # quase instantaneamente e só paga esse custo pesado na primeira
    # chamada de verdade — sessão global cacheada e reaproveitada entre
    # requests depois disso.
    #
    # Modelo "u2netp" (~4.7MB) em vez de "u2net" (~176MB) já ajuda, mas não
    # foi suficiente sozinho: o gráfico de memória do Render (confirmado)
    # mostrava um salto quase vertical até ~100% dos 512MB bem na criação
    # da sessão, INDEPENDENTE do tamanho do modelo — sintoma clássico do
    # arena allocator do onnxruntime, que reserva um bloco grande de
    # memória de antemão por padrão, sem relação com o tamanho real do
    # modelo. `new_session()` do rembg não expõe essas opções, então
    # construímos a sessão diretamente com o arena desligado.
    global _session
    if _session is None:
        import onnxruntime as ort
        from rembg.sessions.u2netp import U2netpSession

        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        _session = U2netpSession("u2netp", sess_opts)
    return _session


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def check_secret(x_shared_secret: Optional[str]):
    if SHARED_SECRET and x_shared_secret != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


class RemoveBackgroundRequest(BaseModel):
    key: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-background")
def remove_background(
    body: RemoveBackgroundRequest,
    x_shared_secret: Optional[str] = Header(default=None),
):
    check_secret(x_shared_secret)

    client = get_r2_client()
    try:
        obj = client.get_object(Bucket=BUCKET, Key=body.key)
        data = obj["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Não achei a imagem no R2: {e}")

    try:
        removed = rembg_process(data, get_session())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao remover fundo: {e}")

    out = BytesIO()
    removed.save(out, format="PNG")
    out.seek(0)

    result_key = f"produtos/{uuid.uuid4()}.png"
    client.put_object(
        Bucket=BUCKET, Key=result_key, Body=out.getvalue(), ContentType="image/png"
    )

    return {"resultKey": result_key}
