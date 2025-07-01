from flask import Blueprint, request, jsonify
from functools import wraps
import time
import aiohttp
import structlog
from api.local_analysis import local_analysis_fallback
from werkzeug.utils import secure_filename
from PIL import Image
import io

logger = structlog.get_logger()
plantnet_bp = Blueprint('plantnet', __name__)

API_CONFIG = {
    "PLANTNET": {
        "url": "https://my-api.plantnet.org/v2/identify/all",
        "project": "all",
        "timeout": 25,
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

def validate_image(file):
    if not file or not secure_filename(file.filename):
        return False, "Invalid filename"
    if not file.content_type.startswith('image/'):
        return False, "Invalid file type"
    try:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > app.config['MAX_CONTENT_LENGTH']:
            return False, "File too large"
        Image.open(file).verify()
        file.seek(0)
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

@plantnet_bp.route("/api/identify-species", methods=["POST"])
@track_request_time
async def identify_species():
    try:
        file = request.files.get("image")
        region = bleach.clean(request.form.get("region", "unknown"), tags=[], strip=True)
        
        is_valid, message = validate_image(file)
        if not is_valid:
            return jsonify({"success": False, "error": message}), 400
        
        image_data = file.read()
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        cache_key = f"species_analysis:hash:{image_hash}:region:{region}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Cache hit for species identification", cache_key=cache_key)
            return jsonify({"success": True, "data": cached_result, "cached": True})
        
        from models import PlantAnalysis, db
        existing_analysis = PlantAnalysis.query.filter_by(image_hash=image_hash, region=region, analysis_type='species').first()
        
        if existing_analysis and existing_analysis.created_at > (datetime.utcnow() - timedelta(hours=24)):
            result = {
                "type": "plant_identification",
                "species": json.loads(existing_analysis.treatment_recommendations),
                "score": existing_analysis.confidence_score,
                "images": []
            }
            cache.set(cache_key, result, timeout=3600)
            return jsonify({"success": True, "data": result, "from_db": True})
        
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('images', image_data, filename='plant.jpg', content_type='image/jpeg')
            form_data.add_field('organs', 'leaf')
            params = {"api-key": app.config['PLANTNET_KEY']}
            timeout = aiohttp.ClientTimeout(total=API_CONFIG["PLANTNET"]["timeout"])
            
            async with session.post(API_CONFIG["PLANTNET"]["url"], params=params, data=form_data, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data.get("results") or len(data["results"]) == 0:
                    raise ValueError("No PlantNet results found")
                
                result = {
                    "type": "plant_identification",
                    "api_source": "plantnet",
                    "species": data["results"][0]["species"],
                    "score": data["results"][0]["score"],
                    "images": data["results"][0].get("images", []),
                }
                
                analysis = PlantAnalysis(
                    image_hash=image_hash,
                    plant_name=data["results"][0]["species"].get("commonNames", ["Unknown"])[0],
                    scientific_name=data["results"][0]["species"]["scientificNameWithoutAuthor"],
                    confidence_score=result["score"],
                    analysis_type="species",
                    treatment_recommendations=json.dumps(result["species"]),
                    region=region,
                    api_source=result["api_source"]
                )
                db.session.add(analysis)
                db.session.commit()
                
                cache.set(cache_key, result, timeout=3600)
                
                logger.info("Species identification successful", species=result["species"]["scientificNameWithoutAuthor"])
                return jsonify({"success": True, "data": result})
    
    except Exception as e:
        logger.error("Species identification error", error=str(e))
        return await local_analysis_fallback(region, image_hash)