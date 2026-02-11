FROM python:3.12-slim

# Set working directory
WORKDIR /app

### Install system dependencies
RUN apt-get update && apt-get install -y \
  netcat-openbsd \
  git \
  wget \
  build-essential \
  zlib1g-dev \
  libbz2-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel "connexion[flask,swagger-ui,uvicorn]"

COPY . /app
RUN pip install /app
COPY requirements.local.txt .
RUN pip install -r requirements.local.txt


# Checkout code
# RUN --mount=type=bind,source=./,target=/app/src/
# COPY \
#   requirements.local.txt \
#   setup.py \
#   pyproject.toml \
#   .
# COPY catch_analysis_tools /app/catch_analysis_tools

# # ARG CAT_DEPLOYMENT
# # RUN if [ "$CAT_DEPLOYMENT" = "prod" ]; then \
# #   git clone git+https://github.com
# #   pip install -r src/requirements.prod.txt; \
# # else \
# # fi

# # Install dependencies
# #RUN pip install -r requirements.local.txt;

EXPOSE 8000

CMD ["python3", "-m", "catch_analysis_tools.app.app"]
