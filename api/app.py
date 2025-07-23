import os
import base64
import requests
import json
import logging
from flask import Flask, request, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import io
import time
import markdown
import bleach
from functools import lru_cache

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, template_folder="../templates")
app.config['UPLOAD_FOLDER'] = '/tmp/Uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB for Vercel
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PERENUAL_API_KEY = os.getenv("PERENUAL_API_KEY")

user_prefs = {"region": None, "language": "en"}
remedy_cache = {}
identification_cache = {}

ALLOWED_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'strong', 'em', 'br', 'div', 'span']
ALLOWED_ATTRIBUTES = {'a': ['href'], 'img': ['src', 'alt']}

def validate_api_key():
    """Validate PLANT_ID_API_KEY with a lightweight test request to /v3/identification."""
    if not PLANT_ID_API_KEY:
        logging.warning("PLANT_ID_API_KEY is missing")
        return False
    try:
        # Create a minimal dummy image (1x1 pixel JPEG)
        img = Image.new('RGB', (1, 1), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_image = f"data:image/jpeg;base64,{encoded}"

        response = requests.post(
            "https://api.plant.id/v3/identification",
            headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
            json={"images": [encoded_image], "similar_images": True},
            timeout=5
        )
        response.raise_for_status()
        logging.info(f"Plant.id API key validated successfully, key: {PLANT_ID_API_KEY[:4]}****")
        return True
    except requests.exceptions.HTTPError as e:
        logging.warning(f"Plant.id API key validation failed: {str(e)}, response: {e.response.text if e.response else 'No response'}")
        return False
    except requests.RequestException as e:
        logging.warning(f"Plant.id API key validation failed: {str(e)}")
        return False

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.site_url = os.getenv("SITE_URL", "http://localhost:5000")
        self.site_name = os.getenv("SITE_NAME", "AgriBot")
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not found, using fallback decision logic")

    def query(self, messages, max_tokens=300, temperature=0.5):
        if not self.api_key:
            return {"error": "OpenRouter API key is missing or invalid.", "model": "none"}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
        for attempt in range(5):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps({
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }),
                    timeout=15
                )
                if response.status_code == 429:
                    wait_time = (2 ** attempt) + (attempt * 1)
                    logging.warning(f"OpenRouter rate limit hit, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                data = response.json()
                if 'choices' not in data:
                    return {"error": "Invalid response from OpenRouter API", "model": "none"}
                selected_model = data.get('model', 'deepseek/deepseek-chat-v3-0324:free')
                logging.info(f"OpenRouter selected model: {selected_model}")
                return {"content": data['choices'][0]['message']['content'].strip(), "model": selected_model}
            except requests.exceptions.Timeout:
                logging.error(f"OpenRouter API timeout after 15 seconds, attempt {attempt + 1}/5")
                if attempt == 4:
                    return {"error": "OpenRouter API timed out after multiple attempts. Please try again later or check your network connection.", "model": "none"}
                wait_time = (2 ** attempt) + (attempt * 1)
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (401, 402, 404):
                    if e.response.status_code == 401:
                        logging.error("Invalid OpenRouter API key")
                        return {"error": "Invalid OpenRouter API key. Please contact support.", "model": "none"}
                    if e.response.status_code == 402:
                        logging.error("OpenRouter API payment required")
                        return {"error": "Payment required for OpenRouter API. Using fallback response.", "model": "none"}
                    if e.response.status_code == 404:
                        logging.error(f"OpenRouter API endpoint not found for model: {self.model}")
                        return {"error": f"OpenRouter API endpoint not found. Check model ({self.model}) or API status.", "model": "none"}
                logging.error(f"OpenRouter API query failed: {str(e)}")
                return {"error": f"OpenRouter API query failed: {str(e)}", "model": "none"}
            except Exception as e:
                logging.error(f"OpenRouter API query failed: {str(e)}")
                return {"error": f"OpenRouter API query failed: {str(e)}", "model": "none"}
        return {"error": "Rate limit exceeded or connection issues. Please try again later. <a href='#' onclick='sendMessage()'>Retry</a>", "model": "none"}

llm_client = LLMClient()

def format_text_for_display(text):
    import re
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002700-\U000027BF"
                               u"\U0001F926-\U0001F937"
                               u"\U00010000-\U0010FFFF"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200D"
                               u"\u23CF"
                               u"\u23E9"
                               u"\u231A"
                               u"\uFE0F"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    html = markdown.markdown(text, extensions=['extra'])
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(file):
    try:
        img = Image.open(file)
        img.verify()
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        return None

@lru_cache(maxsize=100)
def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang or target_lang == "en":
        return text
    system_prompt = f"""
    You are a professional translator. Translate the following text from {source_lang} to {target_lang} accurately, preserving tone and context. Format your response using clean markdown with proper headers (##), bullet points (-), and paragraph breaks for clarity. Ensure the translation is conversational, natural, and well-structured. Do not use emojis or special characters. Return only the translated text.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    data = llm_client.query(messages, max_tokens=1000, temperature=0.3)
    if 'error' in data:
        logging.error(f"Translation failed: {data['error']}")
        return text
    return data['content']

@lru_cache(maxsize=100)
def get_region_specific_remedy(plant_name, scientific_name, disease_name, region, language, plant_id_remedy, disease_description):
    system_prompt = f"""
    You are AgriBot, an AI assistant for Indian agriculture. Plant.id identified the plant as '{plant_name}' (scientific name: {scientific_name}) with disease '{disease_name}'. Plant.id remedy: '{plant_id_remedy}' and disease description: '{disease_description}'. 

    Provide a clear, actionable remedy for {disease_name} in {plant_name} for {region or 'India'} in English. 

    Format your response using clean markdown:
    - Use ## for main sections like "Treatment" or "Prevention"
    - Use ### for subsections
    - Use - for bullet points
    - Use **bold** for important points
    - Use proper paragraph breaks
    - Do not use emojis or special characters

    Consider local weather, cultural practices, and locally available remedies. Include preventive measures and follow-up advice.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Provide a remedy for {disease_name} in {plant_name} in {region or 'India'}."}
    ]
    data = llm_client.query(messages, max_tokens=1000, temperature=0.5)
    if 'error' in data or 'content' not in data:
        logging.error(f"Remedy generation failed: {data.get('error', 'Unknown error')}")
        fallback = f"""
## Treatment for {plant_name} with {disease_name} in {region or 'India'}

### Organic Treatment
- Remove and destroy infected plant parts. **Burn**, **bury deeply**, or **dispose** in garbage; do not compost.
- Apply neem oil or a copper-based organic fungicide, following local agricultural guidelines.

### Chemical Treatment
- Use a fungicide suitable for {disease_name} on {plant_name}. Consult local agricultural extension services for specific recommendations.
- **Follow label instructions** and safety precautions when applying chemicals.

### Preventive Measures
- Ensure proper spacing between plants to improve air circulation.
- Avoid overhead watering to keep foliage dry.
- Regularly inspect plants for early signs of disease.

### Important Notes
- **Monitor plant health** after treatment and reapply as needed.
- Consider the environmental impact of chemical treatments.
- For region-specific advice in {region or 'India'}, consult local agricultural extension services.
"""
        return translate_text(fallback, "en", language), 'none'
    return translate_text(data['content'], "en", language), data['model']

@lru_cache(maxsize=100)
def enhance_identification(plant_name, scientific_name, plant_confidence, disease_name, disease_confidence, region, language, disease_description):
    if plant_confidence < 0.3 or disease_name in ['Fungi', 'Abiotic']:
        system_prompt = f"""
        You are AgriBot, an AI assistant for Indian agriculture. Plant.id identified the plant as '{plant_name}' (scientific name: {scientific_name}) with {plant_confidence:.2%} confidence and disease '{disease_name}' with {disease_confidence:.2%} confidence, description: '{disease_description}'. 

        Refine the plant and disease identification for {region or 'India'} in English, suggesting a more likely plant and specific disease based on regional prevalence, climate, and symptoms. 

        Return the response in JSON format: {{"plant": "...", "scientific_name": "...", "disease": "..."}}.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Refine the identification of {plant_name} and {disease_name} in {region or 'India'}."}
        ]
        data = llm_client.query(messages, max_tokens=500, temperature=0.3)
        if 'error' in data or 'content' not in data:
            return plant_name, scientific_name, plant_confidence, disease_name, disease_confidence
        try:
            refined = json.loads(data['content']) if data['content'].startswith('{') else {
                "plant": plant_name, "scientific_name": scientific_name, "disease": disease_name
            }
            return (
                refined.get('plant', plant_name),
                refined.get('scientific_name', scientific_name or "Unknown"),
                max(plant_confidence, 0.5),
                refined.get('disease', disease_name),
                max(disease_confidence, 0.5)
            )
        except json.JSONDecodeError:
            logging.error("Failed to parse OpenRouter JSON response for identification refinement")
            return plant_name, scientific_name or "Unknown", plant_confidence, disease_name, disease_confidence
    return plant_name, scientific_name or "Unknown", plant_confidence, disease_name, disease_confidence

@lru_cache(maxsize=100)
def perenual_validation(plant_name, region):
    if not PERENUAL_API_KEY:
        logging.warning("PERENUAL_API_KEY not found, skipping validation")
        return plant_name, "Unknown", 0.0
    try:
        response = requests.get(
            f"https://perenual.com/api/species-list?key={PERENUAL_API_KEY}&q={plant_name}",
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        if not data.get('data'):
            return plant_name, "Unknown", 0.0
        for plant in data['data']:
            if plant_name.lower() in plant.get('common_name', '').lower():
                return plant['common_name'], plant.get('scientific_name', ['Unknown'])[0], 0.8
        return plant_name, "Unknown", 0.0
    except requests.RequestException as e:
        logging.error(f"Perenual API query failed: {str(e)}")
        return plant_name, "Unknown", 0.0

@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f"Server error: {str(error)}", exc_info=True)
    response = make_response(jsonify({"error": f"Internal server error. Please try again later. <a href='#' onclick='sendMessage()'>Retry</a>", "typing_effect": False}), 500)
    response.headers['X-RateLimit-Limit'] = 100
    response.headers['X-RateLimit-Remaining'] = 99
    return response

@app.route('/')
def index():
    try:
        response = make_response(render_template('index.html'))
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response
    except Exception as e:
        logging.error(f"Failed to render index.html: {str(e)}")
        response = make_response(jsonify({"error": "Failed to load page. Please try again.", "typing_effect": False}), 500)
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response

@app.route('/favicon.ico')
@app.route('/favicon.png')
def favicon():
    return '', 204

@app.route('/set_prefs', methods=['POST'])
def set_prefs():
    try:
        if 'region' not in request.form or 'language' not in request.form:
            error_msg = "Missing region or language in request."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response
        region = request.form['region'].strip()
        language = request.form['language'].strip()
        if not region or not language:
            error_msg = "Region or language cannot be empty."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response
        user_prefs['region'] = region
        user_prefs['language'] = language
        logging.info(f"User preferences set: region={region}, language={language}")
        response = make_response(jsonify({"status": "success"}))
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response
    except Exception as e:
        logging.error(f"Error setting preferences: {str(e)}")
        response = make_response(jsonify({"error": f"Error setting preferences: {str(e)}. <a href='#' onclick='setPreferences()'>Retry</a>", "typing_effect": False}), 500)
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if 'query' not in request.form:
            error_msg = "No query provided. Please try again."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response
        query = request.form['query'].strip()
        if not query:
            error_msg = "Query cannot be empty. Please try again."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response
        
        query_en = translate_text(query, user_prefs['language'], "en") if user_prefs['language'] != "en" else query
        logging.info(f"Translated query to English: {query_en}")
        
        system_prompt = f"""
        You are AgriBot, an AI assistant for Indian agriculture. Provide clear, concise, and actionable advice for {user_prefs['region'] or 'India'} in English. Consider local weather, cultural practices, and locally available remedies. 

        Format your response using clean markdown:
        - Use ## for main headings
        - Use ### for subheadings
        - Use - for bullet points
        - Use **bold** for emphasis
        - Use proper paragraph breaks
        - Do not use emojis or special characters

        Include detailed advice on soil management, pest control, or irrigation specific to the region. If you cannot provide a specific answer, admit the limitation and suggest consulting local agricultural extension services.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_en}
        ]
        
        data = llm_client.query(messages, max_tokens=1000, temperature=0.5)
        if 'error' in data:
            error_msg = data['error']
            if "Payment required" in error_msg:
                fallback_response = f"""
## Unable to Process Query
Due to API limitations, we cannot process your query right now. Please try the following:

### Recommendations
- Upload an image for plant identification and disease assessment.
- Check your internet connection and try again.
- Contact local agricultural extension services in {user_prefs['region'] or 'India'} for assistance.
"""
                response = make_response(jsonify({
                    "response": format_text_for_display(translate_text(fallback_response, "en", user_prefs['language'])),
                    "model": "none",
                    "typing_effect": False
                }))
                response.headers['X-RateLimit-Limit'] = 100
                response.headers['X-RateLimit-Remaining'] = 99
                return response
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 429)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            response.headers['Retry-After'] = 5
            return response
        
        response_text = translate_text(data['content'], "en", user_prefs['language'])
        formatted_response = format_text_for_display(response_text)
        
        response = make_response(jsonify({
            "response": formatted_response,
            "model": data['model'],
            "typing_effect": False
        }))
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}")
        response = make_response(jsonify({"error": f"Chat error: {str(e)}. <a href='#' onclick='sendMessage()'>Retry</a>", "typing_effect": False}), 500)
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            error_msg = "No image uploaded. Please upload a valid PNG or JPG image."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response
        
        files = request.files.getlist('image')
        if not files or len(files) > 3:
            error_msg = "Please upload 1-3 images (PNG, JPG, JPEG)."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response
        
        encoded_images = []
        for file in files:
            if not file.filename or not allowed_file(file.filename):
                error_msg = f"Invalid file type for {file.filename}. Please upload PNG, JPG, or JPEG."
                response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
                response.headers['X-RateLimit-Limit'] = 100
                response.headers['X-RateLimit-Remaining'] = 99
                return response
            encoded = process_image(file)
            if not encoded:
                error_msg = f"Failed to process image {file.filename}. Please ensure the image is a valid PNG or JPG."
                response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
                response.headers['X-RateLimit-Limit'] = 100
                response.headers['X-RateLimit-Remaining'] = 99
                return response
            encoded_images.append(encoded)

        if not validate_api_key():
            logging.warning(f"Proceeding with identification despite API key validation failure, key: {PLANT_ID_API_KEY[:4]}****")
            error_msg = "Unable to validate Plant.id API key. Attempting identification anyway. If this issue persists, please contact support. <a href='#' onclick='sendMessage()'>Retry</a>"
            # Do not return error immediately; proceed with identification

        plant_id_response = None # Initialize to None
        for attempt in range(3):
            try:
                plant_id_response = requests.post(
                    "https://api.plant.id/v3/identification",
                    headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
                    json={
                        "images": encoded_images,
                        "similar_images": True
                    },
                    timeout=15
                )
                if plant_id_response.status_code == 429:
                    wait_time = (2 ** attempt) + (attempt * 1)
                    logging.warning(f"Plant.id rate limit hit, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                if plant_id_response.status_code == 400:
                    error_details = 'Unknown error'
                    try:
                        if plant_id_response.text:
                            error_details = plant_id_response.json().get('error', 'Unknown error')
                    except json.JSONDecodeError:
                        error_details = plant_id_response.text or "Bad request with no details"

                    logging.error(f"Plant.id identification API 400 error: {error_details}")
                    if "temporary" in str(error_details).lower() and attempt < 2:
                        wait_time = (2 ** attempt) + (attempt * 1)
                        logging.warning(f"Transient 400 error, retrying in {wait_time} seconds")
                        time.sleep(wait_time)
                        continue
                    error_msg = f"Invalid image data or request format: {error_details}. Please upload a valid PNG or JPG image. <a href='#' onclick='sendMessage()'>Retry</a>"
                    response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
                    response.headers['X-RateLimit-Limit'] = 100
                    response.headers['X-RateLimit-Remaining'] = 99
                    return response
                plant_id_response.raise_for_status()
                break # Exit loop on success
            except requests.exceptions.Timeout:
                logging.error(f"Plant.id identification API timeout after 15 seconds, attempt {attempt + 1}/3")
                if attempt == 2:
                    error_msg = "Plant.id API timed out after multiple attempts. Please try again later or check your network connection. <a href='#' onclick='sendMessage()'>Retry</a>"
                    response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
                    response.headers['X-RateLimit-Limit'] = 100
                    response.headers['X-RateLimit-Remaining'] = 99
                    response.headers['Retry-After'] = 5
                    return response
                wait_time = (2 ** attempt) + (attempt * 1)
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                logging.error(f"Plant.id identification API request failed: {str(e)}, response: {e.response.text if e.response else 'No response'}")
                error_msg = f"Plant identification failed: {str(e)}. <a href='#' onclick='sendMessage()'>Retry</a>"
                response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
                response.headers['X-RateLimit-Limit'] = 100
                response.headers['X-RateLimit-Remaining'] = 99
                return response
        
        # Check if response was ever successful
        if plant_id_response is None or not plant_id_response.ok:
            error_msg = "Failed to get a valid response from Plant.id API after several retries."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
            return response

        plant_data = plant_id_response.json()
        if not plant_data.get('result', {}).get('is_plant', {}).get('binary', False):
            error_msg = "The uploaded image does not appear to be a plant. Please upload a valid plant image."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response

        suggestion = plant_data['result']['classification']['suggestions'][0]
        plant_name = suggestion['name'] or "Unknown plant"
        scientific_name = suggestion.get('details', {}).get('scientific_name', 'Unknown')
        plant_confidence = suggestion.get('probability', 0.0)

        disease_resp = None # Initialize to None
        for attempt in range(3):
            try:
                disease_resp = requests.post(
                    "https://api.plant.id/v3/health_assessment",
                    headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
                    json={
                        "images": encoded_images
                    },
                    timeout=15
                )
                if disease_resp.status_code == 429:
                    wait_time = (2 ** attempt) + (attempt * 1)
                    logging.warning(f"Plant.id health assessment rate limit hit, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                if disease_resp.status_code == 400:
                    error_details = 'Unknown error'
                    try:
                        if disease_resp.text:
                            error_details = disease_resp.json().get('error', 'Unknown error')
                    except json.JSONDecodeError:
                        error_details = disease_resp.text or "Bad request with no details"

                    logging.error(f"Plant.id health assessment API 400 error: {error_details}")
                    if "temporary" in str(error_details).lower() and attempt < 2:
                        wait_time = (2 ** attempt) + (attempt * 1)
                        logging.warning(f"Transient 400 error, retrying in {wait_time} seconds")
                        time.sleep(wait_time)
                        continue
                    error_msg = f"Invalid image data or request format: {error_details}. Please upload a valid PNG or JPG image. <a href='#' onclick='sendMessage()'>Retry</a>"
                    response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
                    response.headers['X-RateLimit-Limit'] = 100
                    response.headers['X-RateLimit-Remaining'] = 99
                    return response
                disease_resp.raise_for_status()
                break # Exit loop on success
            except requests.exceptions.Timeout:
                logging.error(f"Plant.id health assessment API timeout after 15 seconds, attempt {attempt + 1}/3")
                if attempt == 2:
                    error_msg = "Plant.id health assessment API timed out after multiple attempts. Please try again later or check your network connection. <a href='#' onclick='sendMessage()'>Retry</a>"
                    response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
                    response.headers['X-RateLimit-Limit'] = 100
                    response.headers['X-RateLimit-Remaining'] = 99
                    response.headers['Retry-After'] = 5
                    return response
                wait_time = (2 ** attempt) + (attempt * 1)
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                logging.error(f"Plant.id health assessment API request failed: {str(e)}, response: {e.response.text if e.response else 'No response'}")
                error_msg = f"Disease assessment failed: {str(e)}. <a href='#' onclick='sendMessage()'>Retry</a>"
                response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
                response.headers['X-RateLimit-Limit'] = 100
                response.headers['X-RateLimit-Remaining'] = 99
                return response
        
        # Check if response was ever successful
        if disease_resp is None or not disease_resp.ok:
            error_msg = "Failed to get a valid response from Plant.id health assessment API after several retries."
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
            return response

        disease_data = disease_resp.json()
        
        # Add detailed logging to debug the response structure
        logging.info(f"Received from Plant.id health assessment: {json.dumps(disease_data, indent=2)}")

        # FIX: Check for the correct keys in the health assessment response
        if not disease_data.get('result', {}).get('is_healthy'):
            error_msg = "Disease assessment failed. Please upload a valid plant image."
            # Log the unexpected structure before returning the error
            logging.error(f"Health assessment keys missing in API response. Data: {disease_data}")
            response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 400)
            response.headers['X-RateLimit-Limit'] = 100
            response.headers['X-RateLimit-Remaining'] = 99
            return response

        if disease_data['result']['is_healthy']['binary']:
            disease_name = "Healthy"
            disease_confidence = disease_data['result']['is_healthy']['probability']
            remedy = f"""
## Plant Health Status for {plant_name} in {user_prefs['region'] or 'India'}

### General Care
- **Watering**: Maintain a consistent schedule, avoiding waterlogging. Adjust based on {user_prefs['region'] or 'local'} climate.
- **Sunlight**: Ensure adequate exposure as per {plant_name} requirements.
- **Ventilation**: Provide good air circulation to prevent fungal growth.

### Maintenance
- **Pruning**: Remove dead or yellowing leaves regularly.
- **Soil**: Use organic compost to enrich soil periodically.
- **Monitoring**: Check soil pH and nutrient levels monthly.

### Regional Considerations
- For {user_prefs['region'] or 'India'}, adapt care to seasonal variations like monsoon or dry periods.
"""
            selected_model = 'none'
        else:
            # FIX: Get disease suggestions from the correct key
            suggestions = disease_data['result']['disease']['suggestions']
            if not suggestions:
                # Handle case where plant is not healthy but no disease is suggested
                disease_name = "Unknown Issue"
                disease_confidence = 1.0 - disease_data['result']['is_healthy']['probability']
                disease_description = "The plant appears to be unhealthy, but a specific disease could not be identified. Please check for common issues like pests, nutrient deficiencies, or watering problems."
                remedy = "## General Advice for Unhealthy Plants\n\n- **Inspect Closely**: Look for signs of pests on the undersides of leaves.\n- **Check Soil Moisture**: Ensure the soil is not too wet or too dry.\n- **Nutrient Check**: Consider applying a balanced fertilizer if the plant shows signs of yellowing."
            else:
                issue = suggestions[0]
                disease_name = issue['name']
                disease_confidence = issue['probability']
                # Details like description and treatment are not provided in this response structure, so we create generic ones.
                disease_description = issue.get('details', {}).get('description', 'No specific description available from the API.')
                treatment_info = issue.get('details', {}).get('treatment', {})
                biological = treatment_info.get('biological', ['Remove and destroy infected parts.'])[0]
                chemical = treatment_info.get('chemical', ['Consult a local expert for chemical treatment options.'])[0]
                remedy = f"""
## Treatment for {plant_name} with {disease_name} in {user_prefs['region'] or 'India'}

### Organic Treatment
- {biological}
- Apply neem oil or copper-based fungicides, following local guidelines.

### Chemical Treatment
- {chemical}
- **Follow label instructions** and safety precautions.

### Preventive Measures
- Ensure proper spacing for air circulation.
- Avoid overhead watering to keep foliage dry.
- Regularly inspect for early disease signs.

### Important Notes
- **Monitor plant health** post-treatment.
- Consider environmental impacts of chemicals.
- Consult local agricultural extension services in {user_prefs['region'] or 'India'} for tailored advice.
"""

        cache_key_id = f"{plant_name}:{scientific_name}:{disease_name}:{user_prefs['region']}:{user_prefs['language']}"
        if cache_key_id in identification_cache:
            plant_name, scientific_name, plant_confidence, disease_name, disease_confidence = identification_cache[cache_key_id]
        else:
            plant_name, scientific_name, plant_confidence, disease_name, disease_confidence = enhance_identification(
                plant_name, scientific_name, plant_confidence, disease_name, disease_confidence, 
                user_prefs['region'], user_prefs['language'], disease_description
            )
            identification_cache[cache_key_id] = (plant_name, scientific_name, plant_confidence, disease_name, disease_confidence)

        cache_key_remedy = f"{plant_name}:{scientific_name}:{disease_name}:{user_prefs['region']}:{user_prefs['language']}"
        if cache_key_remedy in remedy_cache:
            remedy, selected_model = remedy_cache[cache_key_remedy]
            selected_model = 'cached'
        else:
            remedy, selected_model = get_region_specific_remedy(
                plant_name, scientific_name, disease_name, user_prefs['region'], user_prefs['language'], remedy, disease_description
            )
            remedy_cache[cache_key_remedy] = (remedy, selected_model)

        formatted_remedy = format_text_for_display(remedy)

        response = make_response(jsonify({
            "plant": translate_text(plant_name, "en", user_prefs['language']),
            "scientific_name": translate_text(scientific_name, "en", user_prefs['language']),
            "plant_confidence": f"{plant_confidence:.2%}",
            "disease": translate_text(disease_name, "en", user_prefs['language']),
            "disease_confidence": f"{disease_confidence:.2%}",
            "remedy": formatted_remedy,
            "model": selected_model,
            "typing_effect": False
        }))
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        return response
    except Exception as e:
        logging.error(f"Upload endpoint error: {str(e)}", exc_info=True) # Added exc_info for better logging
        error_msg = f"Failed to process image or connect to API: {str(e)}. <a href='#' onclick='sendMessage()'>Retry</a>"
        response = make_response(jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": False}), 500)
        response.headers['X-RateLimit-Limit'] = 100
        response.headers['X-RateLimit-Remaining'] = 99
        response.headers['Retry-After'] = 5
        return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
