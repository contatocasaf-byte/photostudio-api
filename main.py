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
    # considerar o deploy no ar (já aconteceu: o serviço quase foi matado
    # por "timeout de porta" bem na hora que finalmente subia). Adiando pra
    # cá, o servidor abre a porta quase instantaneamente e só paga esse
    # custo pesado na primeira chamada de verdade — sessão global cacheada
    # (modelo u2net, ~170MB) e reaproveitada entre requests depois disso.
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session("u2net")
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
