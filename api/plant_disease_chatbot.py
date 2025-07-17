import os
import base64
import requests
import json
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import io
import time
from functools import lru_cache

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.site_url = os.getenv("SITE_URL", "http://localhost:5000")
        self.site_name = os.getenv("SITE_NAME", "AgriBot")
        self.model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not found, using fallback decision logic")

    def query(self, messages):
        if not self.api_key:
            return {"error": "OpenRouter API key is missing or invalid."}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps({
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 200,
                        "temperature": 0.5
                    })
                )
                response.raise_for_status()
                data = response.json()
                if 'choices' not in data:
                    return {"error": "Invalid response from OpenRouter API"}
                selected_model = data.get('model', 'unknown')
                logging.info(f"Auto Router selected model: {selected_model}")
                return {"content": data['choices'][0]['message']['content'].strip(), "model": selected_model}
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (429, 401, 404):
                    if e.response.status_code == 401:
                        logging.error("Invalid OpenRouter API key")
                        return {"error": "Invalid OpenRouter API key. Please contact support."}
                    if e.response.status_code == 404:
                        logging.error(f"OpenRouter API endpoint not found for model: {self.model}")
                        return {"error": f"OpenRouter API endpoint not found. Check model ({self.model}) or API status."}
                    logging.warning(f"Rate limit hit, retrying in {(2 ** attempt) + (attempt * 0.5)} seconds")
                    time.sleep((2 ** attempt) + (attempt * 0.5))
                    continue
                logging.error(f"OpenRouter API query failed: {str(e)}")
                return {"error": f"OpenRouter API query failed: {str(e)}"}
            except Exception as e:
                logging.error(f"OpenRouter API query failed: {str(e)}")
                return {"error": f"OpenRouter API query failed: {str(e)}"}
        return {"error": "Rate limit exceeded. Please try again later."}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PERENUAL_API_KEY = os.getenv("PERENUAL_API_KEY")
llm_client = LLMClient()

user_prefs = {"region": None, "language": "en"}
remedy_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_prefs', methods=['POST'])
def set_prefs():
    user_prefs['region'] = request.form['region'].strip()
    user_prefs['language'] = request.form['language']
    return jsonify({"status": "success"})

@app.route('/chat', methods=['POST'])
def chat():
    query = request.form['query'].strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    system_prompt = f"You are AgriBot, a helpful AI assistant for Indian agriculture. Answer in {user_prefs['language']} with region-specific advice for {user_prefs['region'] or 'India'}, providing clear, concise, and actionable advice in a friendly tone."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    data = llm_client.query(messages)
    
    if 'error' in data:
        return jsonify({"error": f"Failed to process query: {data['error']} Please try again or use image upload."}), 429
    return jsonify({"response": data['content'], "model": data['model']})

@lru_cache(maxsize=100)
def get_region_specific_remedy(plant_name, disease_name, region, language, plant_id_remedy):
    if not region or language.lower() == 'en':
        if plant_id_remedy and "No biological treatment available" not in plant_id_remedy:
            return plant_id_remedy, 'none'
        system_prompt = f"You are AgriBot, a helpful AI assistant for Indian agriculture. Based on the following Plant.id remedy: '{plant_id_remedy}', provide a clear, concise, and actionable remedy for {disease_name} in {plant_name} for {region or 'India'} in {language}, using a friendly tone and suggesting general plant care practices if no specific treatments are available."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Enhance the remedy for {disease_name} in {plant_name} in {region or 'India'}."}
        ]
        data = llm_client.query(messages)
        if 'error' in data or 'content' not in data:
            return f"{plant_id_remedy}\nFor {region or 'India'}, consult local agricultural extension services.", 'unknown'
        return data['content'], data['model']
    system_prompt = f"You are AgriBot, a helpful AI assistant for Indian agriculture. Based on the following Plant.id remedy: '{plant_id_remedy}', provide a clear, concise, and actionable remedy for {disease_name} in {plant_name} for {region} in {language}, using a friendly tone and suggesting general plant care practices if no specific treatments are available."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Enhance the remedy for {disease_name} in {plant_name} in {region}."}
    ]
    data = llm_client.query(messages)
    if 'error' in data or 'content' not in data:
        return None, 'unknown'
    return data['content'], data['model']

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or JPEG"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        with Image.open(path) as img:
            img.verify()
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        plant_id_response = requests.post(
            "https://api.plant.id/v2/identify",
            headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
            json={
                "images": [encoded],
                "modifiers": ["similar_images"],
                "plant_language": user_prefs['language'],
                "plant_details": ["common_names", "url", "wiki_description"]
            }
        )
        plant_id_response.raise_for_status()
        plant_data = plant_id_response.json()
        if not plant_data.get('suggestions'):
            return jsonify({"error": "Plant identification failed"}), 400
        suggestion = plant_data['suggestions'][0]
        plant_name = suggestion['plant_details']['common_names'][0] if suggestion['plant_details']['common_names'] else suggestion['plant_name'] or "Unknown plant"
        plant_confidence = suggestion['probability']

        disease_resp = requests.post(
            "https://api.plant.id/v2/health_assessment",
            headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
            json={
                "images": [encoded],
                "disease_details": ["common_names", "url", "description", "treatment"]
            }
        )
        disease_resp.raise_for_status()
        disease_data = disease_resp.json()
        if not disease_data.get('health_assessment'):
            return jsonify({"error": "Disease assessment failed"}), 400

        if disease_data['health_assessment']['is_healthy']:
            disease_name = "Healthy"
            remedy = "Your plant appears healthy! Keep up good care practices like proper watering and sunlight."
            disease_confidence = disease_data['health_assessment']['is_healthy_probability']
            selected_model = 'none'
        else:
            issue = disease_data['health_assessment']['diseases'][0]
            disease_name = issue['name']
            disease_confidence = issue['probability']
            treatment_info = issue.get('treatment', {})
            biological = treatment_info.get('biological', ['No biological treatment available'])[0]
            chemical = treatment_info.get('chemical', ['No chemical treatment available'])[0]
            remedy = f"Organic: {biological}\nChemical: {chemical}"

        cache_key = f"{plant_name}:{disease_name}:{user_prefs['region']}:{user_prefs['language']}"
        if cache_key in remedy_cache:
            remedy = remedy_cache[cache_key]
            selected_model = 'cached'
        else:
            enhanced_remedy, selected_model = get_region_specific_remedy(plant_name, disease_name, user_prefs['region'], user_prefs['language'], remedy)
            if enhanced_remedy:
                remedy = enhanced_remedy
                remedy_cache[cache_key] = remedy
            else:
                remedy += f"\nFor {user_prefs['region'] or 'India'}, consult local agricultural extension services."
                selected_model = 'unknown'

        return jsonify({
            "plant": plant_name,
            "plant_confidence": f"{plant_confidence:.2%}",
            "disease": disease_name,
            "disease_confidence": f"{disease_confidence:.2%}",
            "remedy": remedy,
            "model": selected_model
        })
    except (IOError, requests.RequestException) as e:
        return jsonify({"error": f"Failed to process image or connect to API: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)