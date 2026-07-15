FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sem isso, onnxruntime/openmp detectam a contagem de CPUs da máquina
# FÍSICA (não a fatia limitada pelo cgroup do container) e tentam criar um
# thread pool grande demais pra hospedagem — em containers com CPU/RAM
# bem limitadas (ex: Render free/starter) isso trava/mata o processo logo
# na criação da sessão do modelo, mesmo com um modelo pequeno. Forçar 1
# thread evita esse estouro.
ENV OMP_NUM_THREADS=1
ENV OMP_WAIT_POLICY=PASSIVE
ENV OPENBLAS_NUM_THREADS=1

# Render injeta a variável PORT em runtime — o shell exec abaixo permite
# usar essa env var na hora de subir o servidor.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
