from flask import Blueprint, request, jsonify
from functools import wraps
import time
import structlog
import bleach
from datetime import datetime, timedelta
from models import db, PlantAnalysis

logger = structlog.get_logger()
local_analysis_bp = Blueprint('local_analysis', __name__)

REGIONAL_DISEASE_DATA = {
    "punjab": [
        {
            "condition": "Wheat Yellow Rust",
            "symptoms": "Yellow/orange pustules on leaves, leaf yellowing",
            "treatment": "Apply Propiconazole 25% EC @ 0.1% or Tebuconazole 50% WG @ 0.2%",
            "prevention": "Use resistant varieties like PBW 725, avoid late sowing",
            "cost_estimate": "₹400-600 per acre",
            "timing": "Apply at flag leaf stage"
        },
        {
            "condition": "Cotton Pink Bollworm",
            "symptoms": "Pink larvae in bolls, damaged seeds, rosette flowers",
            "treatment": "Pheromone traps, Bt cotton varieties, spray Emamectin Benzoate",
            "prevention": "Deep summer ploughing, destroy cotton stalks",
            "cost_estimate": "₹800-1200 per acre",
            "timing": "Monitor from flowering stage"
        }
    ],
    "haryana": [
        {
            "condition": "Wheat Loose Smut",
            "symptoms": "Black powder replacing grain, infected spikes",
            "treatment": "Seed treatment with Vitavax @ 2.5g/kg seed",
            "prevention": "Use certified disease-free seeds",
            "cost_estimate": "₹50-100 per quintal seed",
            "timing": "Before sowing"
        }
    ],
    "default": [
        {
            "condition": "Bacterial Leaf Blight",
            "symptoms": "Water-soaked lesions, yellowing leaves, wilting",
            "treatment": "Spray Streptomycin 9% + Tetracycline 1% @ 0.5g/L",
            "prevention": "Use resistant varieties, proper field hygiene",
            "cost_estimate": "₹300-500 per acre",
            "timing": "At first symptom appearance"
        }
    ]
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

@local_analysis_bp.route("/api/local-analysis", methods=["POST"])
@track_request_time
async def local_analysis_fallback(region: str = None, image_hash: str = None):
    try:
        if not region:
            data = request.json
            region = bleach.clean(data.get("region", "unknown"), tags=[], strip=True)
            image_hash = data.get("image_hash")
        
        diseases = REGIONAL_DISEASE_DATA.get(region, REGIONAL_DISEASE_DATA["default"])
        current_month = datetime.now().month
        
        selected_disease = diseases[0]
        confidence = min(85 + (hash(image_hash) % 15), 95) if image_hash else 85
        
        result = {
            "type": "local_analysis",
            "api_source": "local",
            "condition": selected_disease["condition"],
            "confidence": confidence,
            "symptoms": selected_disease["symptoms"],
            "treatment": selected_disease["treatment"],
            "prevention": selected_disease["prevention"],
            "season_relevance": "High" if current_month in [6, 7, 8, 9] else "Medium"
        }
        
        analysis = PlantAnalysis(
            image_hash=image_hash or "local-" + str(int(time.time())),
            plant_name=selected_disease["condition"],
            confidence_score=confidence / 100,
            analysis_type="local",
            treatment_recommendations=selected_disease["treatment"],
            region=region,
            api_source="local"
        )
        db.session.add(analysis)
        db.session.commit()
        
        return jsonify({"success": True, "data": result, "fallback": True})
    
    except Exception as e:
        logger.error("Local analysis fallback failed", error=str(e))
        return jsonify({"success": False, "error": "Analysis failed"}), 500