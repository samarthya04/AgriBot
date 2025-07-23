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

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Initialization ---
app = Flask(__name__, template_folder="../templates")
app.config['UPLOAD_FOLDER'] = '/tmp/Uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB Limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- API Keys & Configuration ---
PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PERENUAL_API_KEY = os.getenv("PERENUAL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")

# --- In-memory Caching and User Preferences ---
user_prefs = {"region": None, "language": "en"}
remedy_cache = {}
identification_cache = {}

# --- Security Configuration for HTML Sanitization ---
ALLOWED_TAGS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 
    'strong', 'em', 'br', 'div', 'span', 'hr',
    'table', 'thead', 'tbody', 'tr', 'th', 'td'
]
ALLOWED_ATTRIBUTES = {'a': ['href'], 'img': ['src', 'alt']}


# --- LLM Client for AI-powered Text Generation ---
class LLMClient:
    """A client to interact with the OpenRouter AI API."""
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.site_url = os.getenv("SITE_URL", "http://localhost:5000")
        self.site_name = os.getenv("SITE_NAME", "AgriBot")
        self.model = OPENROUTER_MODEL
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not found, using fallback decision logic.")

    def query(self, messages, temperature=0.5):
        """Sends a query to the OpenRouter API and returns the response."""
        if not self.api_key:
            return {"error": "OpenRouter API key is missing or invalid.", "model": "none"}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        for attempt in range(5):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=20
                )
                if response.status_code == 429:
                    wait_time = (2 ** attempt) + 1
                    logging.warning(f"OpenRouter rate limit hit, retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()

                if 'choices' not in data or not data['choices']:
                    logging.error(f"Invalid response from OpenRouter API: {data}")
                    return {"error": "Invalid response from OpenRouter API.", "model": "none"}
                
                selected_model = data.get('model', self.model)
                logging.info(f"OpenRouter selected model: {selected_model}")
                return {"content": data['choices'][0]['message']['content'].strip(), "model": selected_model}

            except requests.exceptions.Timeout:
                logging.error(f"OpenRouter API timeout, attempt {attempt + 1}/5.")
                if attempt == 4:
                    return {"error": "OpenRouter API timed out. Please try again later.", "model": "none"}
                time.sleep((2 ** attempt) + 1)
            except requests.exceptions.RequestException as e:
                logging.error(f"OpenRouter API query failed: {e}", exc_info=True)
                return {"error": f"OpenRouter API query failed: {e}", "model": "none"}
        
        return {"error": "Rate limit exceeded. Please try again later.", "model": "none"}

llm_client = LLMClient()


# --- Helper Functions ---

def format_text_for_display(text):
    """Sanitizes and formats text for safe HTML display."""
    # The AI can return either Markdown or raw HTML.
    # We let bleach sanitize it to allow safe tags like tables.
    return bleach.clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES)

def allowed_file(filename):
    """Checks if an uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(file_storage):
    """Resizes, converts, and base64-encodes an image file."""
    try:
        img = Image.open(file_storage)
        img.verify()  # Verify that it is, in fact, an image
        # Re-open the file after verification
        file_storage.seek(0)
        img = Image.open(file_storage)
        img = img.convert('RGB')
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logging.error(f"Image processing failed: {e}", exc_info=True)
        return None

@lru_cache(maxsize=100)
def translate_text(text, source_lang, target_lang):
    """Translates text using the LLM, with caching."""
    if source_lang == target_lang or not text:
        return text
        
    system_prompt = f"Translate the following text from {source_lang} to {target_lang}. Return only the translated text."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    
    data = llm_client.query(messages, temperature=0.3)
    return data.get('content', text)

@lru_cache(maxsize=100)
def get_region_specific_remedy(plant_name, scientific_name, disease_name, region, language, plant_id_remedy, disease_description):
    """Generates a region-specific remedy using the LLM."""
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
    data = llm_client.query(messages, temperature=0.5)
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


# --- Flask Routes ---

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler for the Flask app."""
    logging.error(f"An unexpected server error occurred: {error}", exc_info=True)
    response = make_response(jsonify({"error": "Internal server error. Please try again later."}), 500)
    return response

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Handles favicon requests."""
    return '', 204

@app.route('/set_prefs', methods=['POST'])
def set_prefs():
    """Sets user's region and language preferences."""
    region = request.form.get('region', '').strip()
    language = request.form.get('language', '').strip()
    
    if not region or not language:
        return jsonify({"error": "Region and language cannot be empty."}), 400
        
    user_prefs['region'] = region
    user_prefs['language'] = language
    logging.info(f"User preferences set: region={region}, language={language}")
    return jsonify({"status": "success"})

@app.route('/chat', methods=['POST'])
def chat():
    """Handles text-based queries from the user."""
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    query_en = translate_text(query, user_prefs.get('language', 'en'), "en")
    logging.info(f"Translated query to English: {query_en}")
    
    system_prompt = f"You are AgriBot, an AI assistant for Indian agriculture. Provide clear, concise, and actionable advice for {user_prefs.get('region') or 'India'} in English. Format your response using clean markdown or HTML for tables."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query_en}]
    
    data = llm_client.query(messages, temperature=0.5)
    if 'error' in data:
        return jsonify({"error": data['error']}), 500
    
    response_text = translate_text(data['content'], "en", user_prefs.get('language', 'en'))
    formatted_response = format_text_for_display(response_text)
    
    return jsonify({"response": formatted_response, "model": data['model']})

@app.route('/upload', methods=['POST'])
def upload():
    """Handles image uploads for plant identification and health assessment."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    
    files = request.files.getlist('image')
    if not 1 <= len(files) <= 3:
        return jsonify({"error": "Please upload 1-3 images."}), 400
    
    encoded_images = []
    for file in files:
        if not file.filename or not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type: {file.filename}."}), 400
        encoded = process_image(file)
        if not encoded:
            return jsonify({"error": f"Failed to process image: {file.filename}."}), 400
        encoded_images.append(encoded)

    # 1. Plant Identification
    id_payload = {"images": encoded_images, "similar_images": True}
    id_response = _call_plant_id_api("/identification", id_payload)
    if 'error' in id_response:
        return jsonify(id_response), 500
    
    plant_name, scientific_name, plant_confidence = _parse_identification_response(id_response)
    if not plant_name:
         return jsonify({"error": "Could not identify a plant in the image."}), 400

    # 2. Health Assessment
    health_payload = {"images": encoded_images}
    health_response = _call_plant_id_api("/health_assessment", health_payload)
    if 'error' in health_response:
        return jsonify(health_response), 500

    is_healthy, disease_info = _parse_health_response(health_response)
    disease_name = disease_info['name']
    disease_confidence = disease_info['confidence']
    disease_description = disease_info['description']
    
    # 3. Generate Region-Specific Remedy
    remedy, selected_model = get_region_specific_remedy(
        plant_name, scientific_name, disease_name, user_prefs['region'], 
        user_prefs['language'], disease_info['remedy'], disease_description
    )
    
    # 4. Format and Return Final Response
    final_response = {
        "plant": translate_text(plant_name, "en", user_prefs['language']),
        "scientific_name": translate_text(scientific_name, "en", user_prefs['language']),
        "plant_confidence": f"{plant_confidence:.2%}",
        "disease": translate_text(disease_name, "en", user_prefs['language']),
        "disease_confidence": f"{disease_confidence:.2%}",
        "remedy": format_text_for_display(remedy),
        "model": selected_model
    }
    return jsonify(final_response)


# --- Internal Helper Functions for Upload Route ---

def _call_plant_id_api(endpoint, payload):
    """A helper function to call the Plant.id API with retry logic."""
    url = f"https://api.plant.id/v3{endpoint}"
    headers = {"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY}
    
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < 2:
                wait_time = (2 ** attempt) + 1
                logging.warning(f"Plant.id rate limit hit. Retrying in {wait_time}s.")
                time.sleep(wait_time)
            else:
                logging.error(f"Plant.id API request failed: {e}", exc_info=True)
                return {"error": f"Plant.id API error: {e.response.text}"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Plant.id API request failed: {e}", exc_info=True)
            return {"error": "Could not connect to Plant.id service."}
    return {"error": "Plant.id API rate limit exceeded."}

def _parse_identification_response(response_data):
    """Parses the plant identification API response."""
    if not response_data.get('result', {}).get('is_plant', {}).get('binary', False):
        return None, None, 0.0
    
    suggestions = response_data.get('result', {}).get('classification', {}).get('suggestions', [])
    if not suggestions:
        return None, None, 0.0
        
    top_suggestion = suggestions[0]
    plant_name = top_suggestion.get('name', "Unknown plant")
    scientific_name = top_suggestion.get('details', {}).get('scientific_name', 'Unknown')
    plant_confidence = top_suggestion.get('probability', 0.0)
    
    return plant_name, scientific_name, plant_confidence

def _parse_health_response(response_data):
    """Parses the health assessment API response."""
    result = response_data.get('result', {})
    is_healthy_data = result.get('is_healthy', {})
    is_healthy = is_healthy_data.get('binary', False)
    
    disease_info = {
        "name": "Healthy",
        "confidence": is_healthy_data.get('probability', 0.0),
        "description": "The plant appears to be healthy.",
        "remedy": "Maintain current care routine."
    }

    if not is_healthy:
        suggestions = result.get('disease', {}).get('suggestions', [])
        if suggestions:
            top_suggestion = suggestions[0]
            disease_info["name"] = top_suggestion.get('name', "Unknown Issue")
            disease_info["confidence"] = top_suggestion.get('probability', 0.0)
            disease_info["description"] = top_suggestion.get('details', {}).get('description', 'No specific description available.')
        else:
            disease_info["name"] = "Unknown Issue"
            disease_info["confidence"] = 1.0 - is_healthy_data.get('probability', 0.0)
            disease_info["description"] = "The plant seems unhealthy, but no specific disease was identified."

    return is_healthy, disease_info


# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

