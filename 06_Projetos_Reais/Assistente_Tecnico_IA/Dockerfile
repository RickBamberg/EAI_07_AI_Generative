FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 🔹 1. Dependências do sistema (ANTES do pip)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 🔹 2. Copia requirements
COPY requirements.txt .

# 🔹 3. Instala Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 4. Copia o projeto
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]