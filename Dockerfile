# Stage 1: Build the frontend
FROM node:18 AS frontend-builder

WORKDIR /app

RUN mkdir -p /app/public

WORKDIR /app/frontend

# Copy the frontend source code
COPY frontend .

# Install dependencies and build the frontend
RUN npm install
RUN npm run build

# Stage 2: Build the backend
FROM python:3.11-slim AS backend-builder
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Enable venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the backend source code
COPY templates ./templates/
COPY chain.py embedding.py llm.py meta.py server.py main.py ./
COPY requirements.txt .

# Install backend dependencies
RUN pip install -Ur requirements.txt

# Stage 3: Final image
FROM python:3.11-slim

WORKDIR /app

# Copy the built frontend from the frontend-builder stage
COPY --from=frontend-builder /app/public ./public

# Copy the built backend from the backend-builder stage
COPY --from=backend-builder /opt/venv /opt/venv
COPY --from=backend-builder /app .

# Expose the backend port
ENV PATH="/opt/venv/bin:$PATH"
ENV HOST="0.0.0.0"
ENV PORT="80"
ENV LOCAL="0"
ENV ENV="production"
EXPOSE 80

# Start the backend server
CMD ["python3", "main.py"]
