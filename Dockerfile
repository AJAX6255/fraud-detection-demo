# Cloud Run container for Fraud Detection Engine (FastAPI microservice & Streamlit)
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV SERVICE_MODE=api
ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["sh", "-c", "if [ \"$SERVICE_MODE\" = 'streamlit' ]; then streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false; else uvicorn server.main:app --host 0.0.0.0 --port $PORT --workers 4; fi"]
