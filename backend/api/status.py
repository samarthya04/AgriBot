from flask import Blueprint, jsonify
import structlog
import requests
from models import db, APIStatus
from datetime import datetime

logger = structlog.get_logger()
status_bp = Blueprint('status', __name__)

API_CONFIG = {
    "PLANT_ID": {
        "url": "https://plant.id/api/v3/identification",
        "health_url": "https://plant.id/api/v3/health_assessment",
    },
    "PLANTNET": {
        "url": "https://my-api.plantnet.org/v2/identify/all",
    },
    "PERENUAL": {
        "url": "https://perenual.com/api",
    }
}

@status_bp.route("/api/check-status", methods=["GET"])
def check_status():
    statuses = [
        {
            "id": "plant-id-status",
            "name": "Plant.id API",
            "url": f"{API_CONFIG['PLANT_ID']['url']}/health",
            "headers": {"Api-Key": app.config['PLANT_ID_KEY']},
        },
        {
            "id": "plantnet-status",
            "name": "PlantNet API",
            "url": f"{API_CONFIG['PLANTNET']['url']}?api-key={app.config['PLANTNET_KEY']}",
            "headers": {},
        },
        {
            "id": "perenual-status",
            "name": "Perenual API",
            "url": f"{API_CONFIG['PERENUAL']['url']}/health?key={app.config['PERENUAL_KEY']}",
            "headers": {},
        },
    ]

    results = []
    for status in statuses:
        start_time = datetime.utcnow()
        try:
            response = requests.get(status["url"], headers=status["headers"], timeout=5)
            response.raise_for_status()
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            api_status = APIStatus(
                api_name=status["name"].lower().replace(" ", "_"),
                status=True,
                last_checked=datetime.utcnow(),
                response_time_ms=response_time_ms
            )
            db.session.merge(api_status)
            db.session.commit()
            results.append({"id": status["id"], "name": status["name"], "status": "Active"})
        except Exception as e:
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            api_status = APIStatus(
                api_name=status["name"].lower().replace(" ", "_"),
                status=False,
                last_checked=datetime.utcnow(),
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
            db.session.merge(api_status)
            db.session.commit()
            results.append({"id": status["id"], "name": status["name"], "status": "Offline"})
            logger.warning("API status check failed", api=status["name"], error=str(e))
    
    return jsonify(results)