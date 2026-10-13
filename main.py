import os
import threading
import time
import uuid
from io import BytesIO
from typing import Optional

import boto3
from botocore.config import Config
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rembg_logic import rembg_process
import pdf_logic
import pdf_render
from pdf_models import GenerateCatalogPdfRequest, RenderPdfBackgroundRequest

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
    # O arena allocator do onnxruntime reserva um bloco grande de memória
    # de antemão por padrão, independente do tamanho do modelo —
    # `new_session()` do rembg não expõe essas opções, então construímos a
    # sessão diretamente com o arena desligado.
    #
    # Modelo "u2net" (~176MB, melhor qualidade) em vez de "u2netp" (~4.7MB,
    # usado temporariamente enquanto o plano do Render só tinha 512MB de
    # RAM). Medido localmente: pico de ~1000MB com u2net — cabe com folga
    # no plano Standard (2GB). Se a instância voltar a ter pouca RAM no
    # futuro, trocar de volta pra U2netpSession("u2netp", sess_opts) aqui.
    global _session
    if _session is None:
        import onnxruntime as ort
        from rembg.sessions.u2net import U2netSession

        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        _session = U2netSession("u2net", sess_opts)
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


def _build_and_store_pdf(body: GenerateCatalogPdfRequest) -> tuple[str, int]:
    client = get_r2_client()
    pdf_bytes, skipped = pdf_logic.build_pdf(body, client, BUCKET)
    result_key = f"catalogos/pdf/{uuid.uuid4()}.pdf"
    client.put_object(Bucket=BUCKET, Key=result_key, Body=pdf_bytes, ContentType="application/pdf")
    return result_key, skipped


@app.post("/catalogos/pdf")
def generate_catalog_pdf(
    body: GenerateCatalogPdfRequest,
    x_shared_secret: Optional[str] = Header(default=None),
):
    check_secret(x_shared_secret)

    try:
        result_key, skipped = _build_and_store_pdf(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar PDF: {e}")

    return {"resultKey": result_key, "imagensIgnoradas": skipped}


# Geração em segundo plano: a rota da Vercel tem teto de 60s por
# requisição, e catálogo grande passa disso. Aqui o trabalho roda numa
# thread e o cliente consulta o andamento (/status). Estado em memória —
# a instância é única; se o serviço reiniciar no meio do trabalho, o
# status some e o cliente recebe "não encontrado" (basta gerar de novo).
_pdf_jobs: dict[str, dict] = {}
_pdf_jobs_lock = threading.Lock()
PDF_JOB_TTL_SECONDS = 3600


def _run_pdf_job(job_id: str, body: GenerateCatalogPdfRequest) -> None:
    try:
        result_key, skipped = _build_and_store_pdf(body)
        update = {"status": "done", "resultKey": result_key, "imagensIgnoradas": skipped}
    except Exception as e:
        update = {"status": "error", "error": f"Falha ao gerar PDF: {e}"}
    with _pdf_jobs_lock:
        _pdf_jobs[job_id].update(update)


@app.post("/catalogos/pdf/start")
def start_catalog_pdf(
    body: GenerateCatalogPdfRequest,
    x_shared_secret: Optional[str] = Header(default=None),
):
    check_secret(x_shared_secret)

    now = time.time()
    job_id = str(uuid.uuid4())
    with _pdf_jobs_lock:
        for old_id in [k for k, v in _pdf_jobs.items() if now - v["created"] > PDF_JOB_TTL_SECONDS]:
            del _pdf_jobs[old_id]
        _pdf_jobs[job_id] = {"status": "running", "created": now}

    threading.Thread(target=_run_pdf_job, args=(job_id, body), daemon=True).start()
    return {"jobId": job_id}


@app.get("/catalogos/pdf/status/{job_id}")
def catalog_pdf_status(job_id: str, x_shared_secret: Optional[str] = Header(default=None)):
    check_secret(x_shared_secret)

    with _pdf_jobs_lock:
        job = _pdf_jobs.get(job_id)
        result = dict(job) if job else None
    if result is None:
        raise HTTPException(status_code=404, detail="Trabalho não encontrado (o serviço pode ter reiniciado) — gere o PDF de novo.")
    result.pop("created", None)
    return result


@app.post("/catalogos/render-pdf-background")
def render_pdf_background(
    body: RenderPdfBackgroundRequest,
    x_shared_secret: Optional[str] = Header(default=None),
):
    check_secret(x_shared_secret)

    client = get_r2_client()
    try:
        obj = client.get_object(Bucket=BUCKET, Key=body.key)
        pdf_bytes = obj["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Não achei o PDF no R2: {e}")

    try:
        png_bytes = pdf_render.render_pdf_page_to_png(pdf_bytes, body.width, body.height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao converter PDF pra imagem: {e}")

    result_key = f"catalogos/assets/{uuid.uuid4()}.png"
    client.put_object(Bucket=BUCKET, Key=result_key, Body=png_bytes, ContentType="image/png")

    return {"resultKey": result_key}
