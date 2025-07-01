from flask import Blueprint, request, jsonify
from functools import wraps, lru_cache
import time
import requests
import structlog
import bleach
import json

logger = structlog.get_logger()
perenual_bp = Blueprint('perenual', __name__)

API_CONFIG = {
    "PERENUAL": {
        "url": "https://perenual.com/api",
        "timeout": 20,
        "max_retries": 3
    }
}

def track_request_time(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.start_time = time.time()
        try:
            result = f(*args, **kwargs)
            duration_ms = int((time.time() - g.start_time) * 1000)
            logger.info("Request completed", endpoint=request.endpoint, method=request.method, duration_ms=duration_ms, status_code=200)
            return result
        except Exception as e:
            duration_ms = int((time.time() - g.start_time) * 1000)
            logger.error("Request failed", endpoint=request.endpoint, method=request.method, duration_ms=duration_ms, error=str(e))
            raise
    return decorated_function

@perenual_bp.route("/api/plant-info", methods=["POST"])
@track_request_time
@lru_cache(maxsize=1000)
def get_plant_info():
    try:
        query = bleach.clean(request.json.get("query"), tags=[], strip=True)
        region = bleach.clean(request.json.get("region", "unknown"), tags=[], strip=True)
        
        if not query:
            return jsonify({"success": False, "error": "No query provided"}), 400
        
        cache_key = f"plant_info:query:{query}:region:{region}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Cache hit for plant info", cache_key=cache_key)
            return jsonify({"success": True, "data": cached_result, "cached": True})
        
        response = requests.get(
            f"{API_CONFIG['PERENUAL']['url']}/species-list",
            params={"key": app.config['PERENUAL_KEY'], "q": query},
            timeout=API_CONFIG["PERENUAL"]["timeout"]
        )
        response.raise_for_status()
        search_data = response.json()
        
        if not search_data.get("data") or len(search_data["data"]) == 0:
            return jsonify({"success": False, "error": "No plant data found"}), 404
        
        plant = search_data["data"][0]
        detail_response = requests.get(
            f"{API_CONFIG['PERENUAL']['url']}/species/details/{plant['id']}",
            params={"key": app.config['PERENUAL_KEY']},
            timeout=API_CONFIG["PERENUAL"]["timeout"]
        )
        detail_response.raise_for_status()
        detail_data = detail_response.json()
        
        result = {
            "common_name": detail_data.get("common_name", "Unknown"),
            "scientific_name": detail_data.get("scientific_name", ""),
            "family": detail_data.get("family", ""),
            "watering": detail_data.get("watering", ""),
            "sunlight": detail_data.get("sunlight", []),
            "care_level": detail_data.get("care_level", ""),
            "growth_rate": detail_data.get("growth_rate", ""),
        }
        
        cache.set(cache_key, result, timeout=3600)
        
        logger.info("Plant info retrieved", common_name=result["common_name"])
        return jsonify({"success": True, "data": result})
    
    except Exception as e:
        logger.error("Plant info error", error=str(e))
        return jsonify({"success": False, "error": str(e)}), 500