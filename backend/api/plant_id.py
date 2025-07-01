from flask import Blueprint, request, jsonify
from functools import wraps
import time
import aiohttp
import base64
import hashlib
import structlog
from api.local_analysis import local_analysis_fallback

logger = structlog.get_logger()
plant_id_bp = Blueprint('plant_id', __name__)

API_CONFIG = {
    "PLANT_ID": {
        "url": "https://plant.id/api/v3/identification",
        "health_url": "https://plant.id/api/v3/health_assessment",
        "modifiers": ["crops_fast", "similar_images"],
        "plant_details": ["common_names", "url", "description", "treatment", "classification", "cause", "watering", "pruning"],
        "timeout": 30,
        "max_retries": 3
    }
}

def track_request_time(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        g.start_time = time.time()
        try:
            result = await f(*args, **kwargs)
            duration_ms = int((time.time() - g.start_time) * 1000)
            logger.info("Request completed", endpoint=request.endpoint, method=request.method, duration_ms=duration_ms, status_code=200)
            return result
        except Exception as e:
            duration_ms = int((time.time() - g.start_time) * 1000)
            logger.error("Request failed", endpoint=request.endpoint, method=request.method, duration_ms=duration_ms, error=str(e))
            raise
    return decorated_function

@plant_id_bp.route("/api/identify-plant", methods=["POST"])
@track_request_time
async def identify_plant():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        base64_image = data.get("image")
        region = bleach.clean(data.get("region", "unknown"), tags=[], strip=True)
        
        image_data = base64.b64decode(base64_image)
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        cache_key = f"plant_analysis:hash:{image_hash}:region:{region}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Cache hit for plant identification", cache_key=cache_key)
            return jsonify({"success": True, "data": cached_result, "cached": True})
        
        from models import PlantAnalysis, db
        existing_analysis = PlantAnalysis.query.filter_by(image_hash=image_hash, region=region, analysis_type='health').first()
        
        if existing_analysis and existing_analysis.created_at > (datetime.utcnow() - timedelta(hours=24)):
            result = {
                "type": "health_assessment",
                "plant_name": existing_analysis.plant_name,
                "scientific_name": existing_analysis.scientific_name,
                "probability": existing_analysis.confidence_score,
                "diseases": existing_analysis.diseases_detected or [],
                "treatment": existing_analysis.treatment_recommendations
            }
            cache.set(cache_key, result, timeout=3600)
            return jsonify({"success": True, "data": result, "from_db": True})
        
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json", "Api-Key": app.config['PLANT_ID_KEY']}
            payload = {
                "images": [base64_image],
                "modifiers": API_CONFIG["PLANT_ID"]["modifiers"],
                "plant_details": API_CONFIG["PLANT_ID"]["plant_details"],
                "plant_language": "en"
            }
            timeout = aiohttp.ClientTimeout(total=API_CONFIG["PLANT_ID"]["timeout"])
            
            async with session.post(API_CONFIG["PLANT_ID"]["health_url"], headers=headers, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data.get("is_plant") or not data.get("health_assessment"):
                    raise ValueError("Invalid Plant.id response")
                
                result = {
                    "type": "health_assessment",
                    "api_source": "plant_id",
                    "plant_name": data["suggestions"][0]["plant_name"],
                    "scientific_name": data["suggestions"][0].get("plant_details", {}).get("scientific_name", ""),
                    "probability": data["health_assessment"]["probability"],
                    "diseases": data["health_assessment"].get("diseases", []),
                    "details": data["suggestions"][0]["plant_details"],
                }
                
                analysis = PlantAnalysis(
                    image_hash=image_hash,
                    plant_name=result["plant_name"],
                    scientific_name=result.get("scientific_name", ""),
                    confidence_score=result["probability"],
                    analysis_type="health",
                    diseases_detected=result.get("diseases", []),
                    treatment_recommendations=json.dumps(result.get("details", {})),
                    region=region,
                    api_source=result["api_source"]
                )
                db.session.add(analysis)
                db.session.commit()
                
                cache.set(cache_key, result, timeout=3600)
                
                logger.info("Plant identification successful", plant_name=result["plant_name"], confidence=result["probability"])
                return jsonify({"success": True, "data": result})
    
    except Exception as e:
        logger.error("Plant identification error", error=str(e))
        return await local_analysis_fallback(region, image_hash)