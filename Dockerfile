FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

ENV MODELS_DIR=/models 
ENV HF_HOME=${MODELS_DIR} \
    HF_HUB_CACHE=${MODELS_DIR} \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Build arguments for model configuration
ARG RANK=128
ARG LIGHTING=8
ARG USE_ORIGINAL_TEXT_ENCODER="true"

# Make build args available as environment variables
ENV RANK=${RANK} \
    LIGHTING=${LIGHTING} \
    USE_ORIGINAL_TEXT_ENCODER=${USE_ORIGINAL_TEXT_ENCODER}

WORKDIR /app

# Copy application code and requirements
COPY requirements.txt ./
COPY handler.py ./
COPY download_models.py ./
COPY ./models/ ${MODELS_DIR}/

# Install Python dependencies and run model download
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir --no-compile -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
 && python download_models.py ${MODELS_DIR} \
 && find ${MODELS_DIR} -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true \
 && find ${MODELS_DIR} -type f \( -name "*.msgpack" -o -name "*.parquet" -o -name "*.h5" -o -name "*.bin" \) -delete 2>/dev/null || true \
 && find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
 && rm -rf /root/.cache /tmp/* /var/tmp/*

# Expose server port
ENV RP_HANDLER=handler \
    RP_NUM_WORKERS=1 \
    RP_PORT=3000 \
    RP_HTTP=1

EXPOSE 3000

CMD ["python", "handler.py"]