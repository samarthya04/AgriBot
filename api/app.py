import os
import base64
import requests
import json
import logging
from functools import wraps
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, g
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import io
import time
import markdown
import bleach
from upstash_redis import Redis
from hashlib import md5
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agribot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__, template_folder="../templates")
app.config.update(
    UPLOAD_FOLDER='/tmp/uploads',
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5MB
    SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False
)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Redis client initialization with error handling
try:
    redis_client = Redis(
        url=os.getenv("UPSTASH_REDIS_REST_URL"),
        token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
    )
    # Test connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# API configuration
PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PERENUAL_API_KEY = os.getenv("PERENUAL_API_KEY")

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Enhanced configuration
class Config:
    MAX_RETRIES = 3
    TIMEOUT = 10
    CACHE_TTL = {
        'short': 30,
        'medium': 300,
        'long': 3600
    }
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_IMAGE_SIZE = (800, 600)
    IMAGE_QUALITY = 85

config = Config()

# HTML sanitization
ALLOWED_TAGS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 
    'strong', 'em', 'br', 'div', 'span', 'blockquote', 'code', 'pre'
]
ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
    'img': ['src', 'alt', 'title'],
    'div': ['class'],
    'span': ['class']
}

# User preferences with validation
class UserPreferences:
    def __init__(self):
        self.region = None
        self.language = "en"
        self.supported_languages = ["en", "hi", "bn", "te", "ta", "ml", "kn", "gu", "mr", "pa"]
    
    def validate_language(self, lang):
        return lang if lang in self.supported_languages else "en"
    
    def set_preferences(self, region, language):
        self.region = region.strip() if region else None
        self.language = self.validate_language(language)

user_prefs = UserPreferences()

# Decorators
def handle_exceptions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                "error": "An unexpected error occurred. Please try again.",
                "typing_effect": True
            }), 500
    return decorated_function

def rate_limit(max_requests=10, window=60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not redis_client:
                return f(*args, **kwargs)
            
            client_ip = request.remote_addr
            key = f"rate_limit:{client_ip}:{f.__name__}"
            
            try:
                current = redis_client.get(key)
                if current and int(current) >= max_requests:
                    return jsonify({
                        "error": "Rate limit exceeded. Please try again later.",
                        "typing_effect": True
                    }), 429
                
                pipe = redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window)
                pipe.execute()
                
            except Exception as e:
                logger.warning(f"Rate limiting error: {e}")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Enhanced caching utilities
class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}
        self.cache_lock = threading.Lock()
    
    def get(self, key):
        if not self.redis:
            return self.local_cache.get(key)
        
        try:
            cached = self.redis.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return self.local_cache.get(key)
    
    def set(self, key, value, ttl=300):
        if not self.redis:
            with self.cache_lock:
                self.local_cache[key] = value
                # Simple LRU: keep only last 100 items
                if len(self.local_cache) > 100:
                    oldest_key = next(iter(self.local_cache))
                    del self.local_cache[oldest_key]
            return
        
        try:
            self.redis.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            with self.cache_lock:
                self.local_cache[key] = value

cache_manager = CacheManager(redis_client)

# Enhanced LLM Client
class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.site_url = os.getenv("SITE_URL", "http://localhost:5000")
        self.site_name = os.getenv("SITE_NAME", "AgriBot")
        self.model = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found")
    
    def _make_request(self, messages, max_tokens=500, temperature=0.5):
        """Make API request with proper error handling"""
        if not self.api_key:
            raise ValueError("API key not available")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        for attempt in range(config.MAX_RETRIES):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=config.TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif e.response.status_code in [401, 402, 403]:
                    raise ValueError(f"API authentication error: {e.response.status_code}")
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise requests.exceptions.RequestException("Max retries exceeded")
    
    def query(self, messages, max_tokens=500, temperature=0.5):
        """Query LLM with caching"""
        cache_key = f"llm:{md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()}"
        
        # Check cache
        cached_response = cache_manager.get(cache_key)
        if cached_response:
            logger.info("Returning cached LLM response")
            return cached_response
        
        try:
            response_data = self._make_request(messages, max_tokens, temperature)
            
            if 'choices' not in response_data or not response_data['choices']:
                raise ValueError("Invalid API response structure")
            
            result = {
                "content": response_data['choices'][0]['message']['content'].strip(),
                "model": response_data.get('model', self.model),
                "usage": response_data.get('usage', {})
            }
            
            # Cache successful response
            cache_manager.set(cache_key, result, config.CACHE_TTL['short'])
            return result
            
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            return {
                "error": str(e),
                "model": "none"
            }

llm_client = LLMClient()

# Enhanced image processing
def process_image(file_storage):
    """Process uploaded image with enhanced error handling"""
    try:
        # Read and validate image
        file_storage.seek(0)
        img = Image.open(file_storage)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Resize intelligently
        img.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=config.IMAGE_QUALITY, optimize=True)
        
        # Encode to base64
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Image processed: {img.size}, size: {len(encoded)} chars")
        return encoded
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS)

# Enhanced text processing
def format_text_for_display(text):
    """Format text with improved markdown processing"""
    try:
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Convert markdown to HTML
        html = markdown.markdown(
            text, 
            extensions=['extra', 'codehilite', 'toc'],
            extension_configs={
                'codehilite': {'css_class': 'highlight'},
                'toc': {'permalink': True}
            }
        )
        
        # Sanitize HTML
        clean_html = bleach.clean(
            html, 
            tags=ALLOWED_TAGS, 
            attributes=ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        return clean_html
        
    except Exception as e:
        logger.error(f"Text formatting failed: {str(e)}")
        return text

# Enhanced translation
def translate_text(text, source_lang, target_lang):
    """Translate text with caching and fallback"""
    if source_lang == target_lang or target_lang == "en":
        return text
    
    cache_key = f"translate:{md5((text + source_lang + target_lang).encode()).hexdigest()}"
    cached_translation = cache_manager.get(cache_key)
    
    if cached_translation:
        logger.info("Returning cached translation")
        return cached_translation
    
    system_prompt = f"""
    You are a professional translator specializing in agricultural and technical content.
    Translate the following text from {source_lang} to {target_lang} accurately.
    Preserve technical terms, maintain formatting, and ensure cultural appropriateness.
    Return only the translated text.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    result = llm_client.query(messages, max_tokens=1000, temperature=0.3)
    
    if 'error' in result:
        logger.error(f"Translation failed: {result['error']}")
        return text
    
    translation = result['content']
    cache_manager.set(cache_key, translation, config.CACHE_TTL['long'])
    return translation

# API integration utilities
class PlantIDClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.plant.id/v2"
    
    def identify_plant(self, images):
        """Identify plant from images"""
        if not self.api_key:
            raise ValueError("Plant.id API key not available")
        
        response = requests.post(
            f"{self.base_url}/identify",
            headers={
                "Content-Type": "application/json",
                "Api-Key": self.api_key
            },
            json={
                "images": images,
                "modifiers": ["similar_images", "crops_fast"],
                "plant_language": "en",
                "plant_details": [
                    "common_names", "url", "wiki_description", 
                    "taxonomy", "edible_parts", "propagation_methods"
                ]
            },
            timeout=config.TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    
    def assess_health(self, images):
        """Assess plant health from images"""
        if not self.api_key:
            raise ValueError("Plant.id API key not available")
        
        response = requests.post(
            f"{self.base_url}/health_assessment",
            headers={
                "Content-Type": "application/json",
                "Api-Key": self.api_key
            },
            json={
                "images": images,
                "disease_details": [
                    "common_names", "url", "description", 
                    "treatment", "classification", "cause"
                ]
            },
            timeout=config.TIMEOUT
        )
        response.raise_for_status()
        return response.json()

plant_id_client = PlantIDClient(PLANT_ID_API_KEY)

# Enhanced route handlers
@app.before_request
def before_request():
    """Pre-request setup"""
    g.start_time = time.time()

@app.after_request
def after_request(response):
    """Post-request logging"""
    duration = time.time() - g.start_time
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large. Maximum size is 5MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.route('/')
def index():
    """Serve main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Failed to render index.html: {str(e)}")
        return jsonify({"error": "Failed to load page"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    })

@app.route('/set_prefs', methods=['POST'])
@handle_exceptions
@rate_limit(max_requests=5, window=60)
def set_preferences():
    """Set user preferences"""
    try:
        data = request.get_json() or request.form
        region = data.get('region', '').strip()
        language = data.get('language', 'en')
        
        user_prefs.set_preferences(region, language)
        
        logger.info(f"User preferences set: region={region}, language={language}")
        return jsonify({
            "status": "success",
            "region": user_prefs.region,
            "language": user_prefs.language
        })
        
    except Exception as e:
        logger.error(f"Error setting preferences: {str(e)}")
        return jsonify({"error": "Failed to set preferences"}), 400

@app.route('/chat', methods=['POST'])
@handle_exceptions
@rate_limit(max_requests=20, window=60)
def chat():
    """Handle chat queries"""
    try:
        data = request.get_json() or request.form
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                "error": "Please provide a query",
                "typing_effect": True
            }), 400
        
        # Check cache
        cache_key = f"chat:{md5(query.encode()).hexdigest()}"
        cached_response = cache_manager.get(cache_key)
        if cached_response:
            logger.info("Returning cached chat response")
            return jsonify(cached_response)
        
        # Translate query if needed
        query_en = (translate_text(query, user_prefs.language, "en") 
                   if user_prefs.language != "en" else query)
        
        # Prepare system prompt
        system_prompt = f"""
        You are AgriBot, an AI assistant for Indian agriculture. Provide clear, actionable advice for farming in {user_prefs.region or 'India'}.
        
        Guidelines:
        - Use simple, clear language
        - Provide specific, actionable recommendations
        - Consider local climate and soil conditions
        - Include traditional and modern farming techniques
        - Use proper markdown formatting
        - Be concise but comprehensive
        
        Format your response with:
        - ## for main headings
        - ### for subheadings  
        - - for bullet points
        - **bold** for emphasis
        - Proper paragraph breaks
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_en}
        ]
        
        # Get LLM response
        result = llm_client.query(messages, max_tokens=800, temperature=0.6)
        
        if 'error' in result:
            return jsonify({
                "error": f"Unable to process query: {result['error']}",
                "typing_effect": True
            }), 500
        
        # Translate response if needed
        response_text = (translate_text(result['content'], "en", user_prefs.language)
                        if user_prefs.language != "en" else result['content'])
        
        # Format response
        formatted_response = format_text_for_display(response_text)
        
        response_obj = {
            "response": formatted_response,
            "model": result['model'],
            "typing_effect": True
        }
        
        # Cache response
        cache_manager.set(cache_key, response_obj, config.CACHE_TTL['short'])
        
        return jsonify(response_obj)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            "error": "Failed to process chat query",
            "typing_effect": True
        }), 500

@app.route('/upload', methods=['POST'])
@handle_exceptions
@rate_limit(max_requests=10, window=300)
def upload_images():
    """Handle image upload and analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({
                "error": "No images uploaded",
                "typing_effect": True
            }), 400
        
        files = request.files.getlist('image')
        
        if not files or len(files) > 3:
            return jsonify({
                "error": "Please upload 1-3 images",
                "typing_effect": True
            }), 400
        
        # Process images
        encoded_images = []
        for file in files:
            if not file.filename or not allowed_file(file.filename):
                return jsonify({
                    "error": f"Invalid file type: {file.filename}",
                    "typing_effect": True
                }), 400
            
            encoded = process_image(file)
            if not encoded:
                return jsonify({
                    "error": f"Failed to process image: {file.filename}",
                    "typing_effect": True
                }), 400
            
            encoded_images.append(encoded)
        
        # Check cache
        cache_key = f"upload:{md5(''.join(encoded_images).encode()).hexdigest()}"
        cached_response = cache_manager.get(cache_key)
        if cached_response:
            logger.info("Returning cached upload response")
            return jsonify(cached_response)
        
        # Plant identification
        try:
            plant_data = plant_id_client.identify_plant(encoded_images)
            if not plant_data.get('suggestions'):
                return jsonify({
                    "error": "Unable to identify plant",
                    "typing_effect": True
                }), 400
            
            suggestion = plant_data['suggestions'][0]
            plant_details = suggestion.get('plant_details', {})
            
            plant_name = (plant_details.get('common_names', ['Unknown'])[0] 
                         if plant_details.get('common_names') else 'Unknown')
            scientific_name = plant_details.get('scientific_name', 'Unknown')
            plant_confidence = suggestion.get('probability', 0.0)
            
        except Exception as e:
            logger.error(f"Plant identification failed: {str(e)}")
            return jsonify({
                "error": "Plant identification service unavailable",
                "typing_effect": True
            }), 500
        
        # Health assessment
        try:
            health_data = plant_id_client.assess_health(encoded_images)
            health_assessment = health_data.get('health_assessment', {})
            
            if health_assessment.get('is_healthy', False):
                disease_name = "Healthy"
                disease_confidence = health_assessment.get('is_healthy_probability', 0.0)
                remedy = generate_healthy_plant_advice(plant_name, user_prefs.region)
            else:
                diseases = health_assessment.get('diseases', [])
                if diseases:
                    disease = diseases[0]
                    disease_name = disease.get('name', 'Unknown disease')
                    disease_confidence = disease.get('probability', 0.0)
                    remedy = generate_disease_treatment(
                        plant_name, scientific_name, disease_name, 
                        disease.get('disease_details', {}), user_prefs.region
                    )
                else:
                    disease_name = "Unknown condition"
                    disease_confidence = 0.0
                    remedy = "Unable to determine specific treatment. Please consult a local agricultural expert."
            
        except Exception as e:
            logger.error(f"Health assessment failed: {str(e)}")
            disease_name = "Assessment unavailable"
            disease_confidence = 0.0
            remedy = "Health assessment service unavailable. Please try again later."
        
        # Format response
        formatted_remedy = format_text_for_display(remedy)
        
        response_obj = {
            "plant": translate_text(plant_name, "en", user_prefs.language),
            "scientific_name": translate_text(scientific_name, "en", user_prefs.language),
            "plant_confidence": f"{plant_confidence:.1%}",
            "disease": translate_text(disease_name, "en", user_prefs.language),
            "disease_confidence": f"{disease_confidence:.1%}",
            "remedy": translate_text(formatted_remedy, "en", user_prefs.language),
            "typing_effect": True
        }
        
        # Cache response
        cache_manager.set(cache_key, response_obj, config.CACHE_TTL['medium'])
        
        return jsonify(response_obj)
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        return jsonify({
            "error": "Failed to process images",
            "typing_effect": True
        }), 500

def generate_healthy_plant_advice(plant_name, region):
    """Generate advice for healthy plants"""
    return f"""
## Plant Health Status: Excellent! 🌱

Your {plant_name} appears to be in good health. Here's how to maintain its vitality:

### Care Guidelines
- **Watering**: Maintain consistent moisture without waterlogging
- **Light**: Ensure adequate sunlight for the species
- **Nutrients**: Regular feeding with balanced fertilizer
- **Pruning**: Remove dead or damaged parts regularly

### Regional Considerations for {region or 'India'}
- Monitor seasonal changes and adjust care accordingly
- Be aware of local pest seasons
- Consider monsoon and dry season requirements

### Preventive Measures
- Regular inspection for early problem detection
- Proper spacing for air circulation
- Soil health maintenance with organic matter
"""

def generate_disease_treatment(plant_name, scientific_name, disease_name, disease_details, region):
    """Generate treatment recommendations"""
    description = disease_details.get('description', 'No description available')
    treatment = disease_details.get('treatment', {})
    
    return f"""
## Disease Identified: {disease_name}

### Description
{description}

### Treatment Options

#### Organic Treatment
{treatment.get('biological', ['Organic treatment methods recommended'])[0]}

#### Chemical Treatment
{treatment.get('chemical', ['Chemical treatment may be necessary'])[0]}

### Regional Considerations for {region or 'India'}
- Consider local climate conditions
- Use treatments appropriate for the season
- Follow local agricultural guidelines

### Prevention
- Improve air circulation around plants
- Avoid overhead watering
- Remove affected plant material
- Maintain proper soil drainage

### When to Seek Help
Contact your local agricultural extension office if symptoms persist or worsen.
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting AgriBot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
