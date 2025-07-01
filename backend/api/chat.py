from flask import Blueprint, request, jsonify
from functools import wraps
import time
import structlog
import bleach
from openai import OpenAI
from models import db, Query
import json

logger = structlog.get_logger()
chat_bp = Blueprint('chat', __name__)

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

@chat_bp.route("/api/chat", methods=["POST"])
@track_request_time
def enhanced_chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"success": False, "error": "No message provided"}), 400
        
        message = bleach.clean(data.get("message"), tags=['p', 'br', 'strong'], strip=True, max_length=500)
        region = bleach.clean(data.get("region", "unknown"), tags=[], strip=True)
        language = bleach.clean(data.get("language", "english"), tags=[], strip=True)
        
        if not message:
            return jsonify({"success": False, "error": "Empty message"}), 400
        
        cache_key = f"chat:message:{message}:region:{region}:language:{language}"
        cached_response = cache.get(cache_key)
        if cached_response:
            return jsonify({"success": True, "response": cached_response, "cached": True})
        
        openai_client = OpenAI(api_key=app.config['OPENAI_API_KEY'], http_client=None)
        system_prompt = f"""
        You are AgriBot Enhanced, an expert AI agricultural assistant specifically designed for Indian farmers in {region}.
        
        CONTEXT:
        - Region: {region} (provide region-specific advice based on local climate, soil, and crops)
        - User Language: {language} (incorporate local terms when helpful)
        - Knowledge Base: ICAR guidelines, state agriculture department recommendations, modern farming practices
        
        GUIDELINES:
        1. Provide accurate, actionable agricultural advice
        2. Consider local climate, soil conditions, and crop patterns for {region}
        3. Include specific product recommendations when relevant (fertilizers, pesticides, tools)
        4. Mention government schemes or subsidies available in {region} when applicable
        5. Use conversational, farmer-friendly language
        6. Include seasonal timing recommendations
        7. Always prioritize sustainable and organic methods when possible
        
        RESPONSE FORMAT:
        - Start with a direct answer
        - Provide step-by-step guidance when needed
        - Include specific quantities/dosages
        - Mention costs when relevant
        - Add prevention tips
        
        USER QUERY: {message}
        """
        
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=600,
            temperature=0.7
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        reply = response.choices[0].message.content.strip()
        
        query = Query(
            message=message,
            response=reply,
            region=region,
            language=language,
            response_time_ms=response_time_ms
        )
        db.session.add(query)
        db.session.commit()
        
        cache.set(cache_key, reply, timeout=1800)
        
        logger.info("Chat response generated", message_length=len(message), response_time_ms=response_time_ms)
        return jsonify({"success": True, "response": reply})
    
    except Exception as e:
        logger.error("Chat error", error=str(e))
        return jsonify({"success": False, "error": "Failed to generate response"}), 500