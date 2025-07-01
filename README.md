# AgriBot Enhanced

A Flask-based AI-powered agricultural assistant for Indian farmers, integrating multiple plant identification APIs, region-specific advice, and a responsive frontend.

## Setup Instructions

1. **Prerequisites**:
   - Python 3.9+
   - Redis (`sudo apt-get install redis-server` or Windows equivalent)
   - SQLite (included with Python)

2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r backend/requirements.txt
   ```

3. **Configure Environment**:
   - Copy `backend/.env` template and fill in API keys:
     ```
     SECRET_KEY=$(python -c "import os; print(os.urandom(24).hex())")
     DATABASE_URL=sqlite:///agribot.db
     REDIS_URL=redis://localhost:6379/0
     PLANT_ID_KEY=your_plant_id_key_here
     PLANTNET_KEY=your_plantnet_key_here
     PERENUAL_KEY=your_perenual_key_here
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Initialize Database**:
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

5. **Run Application**:
   ```bash
   python backend/app.py
   ```
   Access at `http://localhost:5000`.

6. **Production Deployment**:
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 4 --worker-class gevent --worker-connections 1000 --timeout 30 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100 --log-level info --access-logfile - --error-logfile - backend.app:app
   ```

7. **Docker Deployment**:
   ```bash
   docker build -t agribot .
   docker run -p 5000:5000 --env-file backend/.env agribot
   ```

## Testing
- **Image Upload**: Upload a plant image to test disease/species identification.
- **Chat**: Query "best crops for Punjab" to verify region-specific advice.
- **API Status**: Check status indicators in the UI.
- **Logs**: Review `backend/logs/app.log` for structured logs.
- **Endpoints**:
  - `POST /api/upload-image`: Upload image
  - `POST /api/identify-plant`: Plant health analysis
  - `POST /api/identify-species`: Species identification
  - `POST /api/plant-info`: Plant details
  - `POST /api/chat`: AI chat
  - `POST /api/local-analysis`: Local fallback
  - `GET /api/check-status`: API status
  - `GET /health`: System health

## Troubleshooting
- **OpenAI Error**: Ensure `openai==1.10.0` and valid `OPENAI_API_KEY`.
- **Redis**: Verify Redis is running (`redis-cli ping` should return `PONG`).
- **Database**: Run `flask db upgrade` if schema errors occur.
- **Logs**: Check `backend/logs/app.log` for detailed errors.