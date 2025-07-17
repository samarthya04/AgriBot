# Use official Python slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ api/
COPY templates/ templates/

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Command to run the app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api.app:app"]
