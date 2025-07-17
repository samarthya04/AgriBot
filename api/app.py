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
import markdown
import bleach
from functools import lru_cache

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.site_url = os.getenv("SITE_URL", "http://localhost:5000")
        self.site_name = os.getenv("SITE_NAME", "AgriBot")
        # Use the specified free model
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
        for attempt in range(3):
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
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                if 'choices' not in data:
                    return {"error": "Invalid response from OpenRouter API", "model": "none"}
                selected_model = data.get('model', 'deepseek/deepseek-chat-v3-0324:free')
                logging.info(f"OpenRouter selected model: {selected_model}")
                return {"content": data['choices'][0]['message']['content'].strip(), "model": selected_model}
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (429, 401, 402, 404):
                    if e.response.status_code == 401:
                        logging.error("Invalid OpenRouter API key")
                        return {"error": "Invalid OpenRouter API key. Please contact support.", "model": "none"}
                    if e.response.status_code == 402:
                        logging.error("OpenRouter API payment required")
                        return {"error": "Payment required for OpenRouter API. Using fallback response.", "model": "none"}
                    if e.response.status_code == 404:
                        logging.error(f"OpenRouter API endpoint not found for model: {self.model}")
                        return {"error": f"OpenRouter API endpoint not found. Check model ({self.model}) or API status.", "model": "none"}
                    logging.warning(f"Rate limit hit, retrying in {(2 ** attempt) + (attempt * 0.5)} seconds")
                    time.sleep((2 ** attempt) + (attempt * 0.5))
                    continue
                logging.error(f"OpenRouter API query failed: {str(e)}")
                return {"error": f"OpenRouter API query failed: {str(e)}", "model": "none"}
            except Exception as e:
                logging.error(f"OpenRouter API query failed: {str(e)}")
                return {"error": f"OpenRouter API query failed: {str(e)}", "model": "none"}
        return {"error": "Rate limit exceeded. Please try again later.", "model": "none"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PERENUAL_API_KEY = os.getenv("PERENUAL_API_KEY")
llm_client = LLMClient()

user_prefs = {"region": None, "language": "en"}
remedy_cache = {}
identification_cache = {}

ALLOWED_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'strong', 'em', 'br', 'div', 'span']
ALLOWED_ATTRIBUTES = {'a': ['href'], 'img': ['src', 'alt']}

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
    return convert_markdown_to_html(text)

def convert_markdown_to_html(text):
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
        img = img.resize((800, 600), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        return None

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_prefs', methods=['POST'])
def set_prefs():
    user_prefs['region'] = request.form['region'].strip()
    user_prefs['language'] = request.form['language']
    logging.info(f"User preferences set: region={user_prefs['region']}, language={user_prefs['language']}")
    return jsonify({"status": "success"})

@app.route('/chat', methods=['POST'])
def chat():
    query = request.form['query'].strip()
    if not query:
        error_msg = "No query provided. Please try again."
        return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400
    
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
    - Be conversational and friendly
    - Do not use emojis or special characters

    Include detailed advice on soil management, pest control, or irrigation specific to the region.
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
- Upload an image for plant identification and disease assessment.
- Check your internet connection and try again.
- Contact local agricultural extension services in {user_prefs['region'] or 'India'} for assistance.
            """
            return jsonify({
                "response": format_text_for_display(translate_text(fallback_response, "en", user_prefs['language'])),
                "model": "none",
                "typing_effect": True
            })
        return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 429
    
    response_text = translate_text(data['content'], "en", user_prefs['language'])
    formatted_response = format_text_for_display(response_text)
    
    logging.info(f"Formatted response for {user_prefs['language']}")
    return jsonify({
        "response": formatted_response,
        "model": data['model'],
        "typing_effect": True
    })

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
        fallback = f"""
## Treatment
{plant_id_remedy}

## Note
For {region or 'India'}, consult local agricultural extension services for region-specific advice due to API limitations.
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
                refined.get('scientific_name', scientific_name),
                max(plant_confidence, 0.5),
                refined.get('disease', disease_name),
                max(disease_confidence, 0.5)
            )
        except json.JSONDecodeError:
            logging.error("Failed to parse OpenRouter JSON response for identification refinement")
            return plant_name, scientific_name, plant_confidence, disease_name, disease_confidence
    return plant_name, scientific_name, plant_confidence, disease_name, disease_confidence

@lru_cache(maxsize=100)
def perenual_validation(plant_name, region):
    if not PERENUAL_API_KEY:
        logging.warning("PERENUAL_API_KEY not found, skipping validation")
        return plant_name, "", 0.0
    try:
        response = requests.get(
            f"https://perenual.com/api/species-list?key={PERENUAL_API_KEY}&q={plant_name}",
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if not data.get('data'):
            return plant_name, "", 0.0
        for plant in data['data']:
            if plant_name.lower() in plant.get('common_name', '').lower():
                return plant['common_name'], plant.get('scientific_name', [''])[0], 0.8
        return plant_name, "", 0.0
    except requests.RequestException as e:
        logging.error(f"Perenual API query failed: {str(e)}")
        return plant_name, "", 0.0

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        error_msg = "No image uploaded."
        return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400
    
    files = request.files.getlist('image')
    if not files or len(files) > 3:
        error_msg = "Please upload 1-3 images (PNG, JPG, JPEG)."
        return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400
    
    encoded_images = []
    for file in files:
        if not file.filename or not allowed_file(file.filename):
            error_msg = f"Invalid file type for {file.filename}. Please upload PNG, JPG, or JPEG."
            return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400
        encoded = process_image(file)
        if not encoded:
            error_msg = f"Failed to process image {file.filename}."
            return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400
        encoded_images.append(encoded)

    try:
        plant_id_response = requests.post(
            "https://api.plant.id/v2/identify",
            headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
            json={
                "images": encoded_images,
                "modifiers": ["similar_images", "crops_fast"],
                "plant_language": "en",
                "plant_details": ["common_names", "url", "wiki_description", "taxonomy", "edible_parts"]
            },
            timeout=10
        )
        plant_id_response.raise_for_status()
        plant_data = plant_id_response.json()
        if not plant_data.get('suggestions'):
            error_msg = "Plant identification failed."
            return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400
        suggestion = plant_data['suggestions'][0]
        plant_name = suggestion['plant_details']['common_names'][0] if suggestion['plant_details']['common_names'] else suggestion['plant_name'] or "Unknown plant"
        scientific_name = suggestion['plant_details'].get('scientific_name', 'Unknown scientific name')
        plant_confidence = suggestion['probability']

        if plant_confidence < 0.3 and PERENUAL_API_KEY:
            plant_name, scientific_name, perenual_confidence = perenual_validation(plant_name, user_prefs['region'])
            plant_confidence = max(plant_confidence, perenual_confidence)

        disease_resp = requests.post(
            "https://api.plant.id/v2/health_assessment",
            headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY},
            json={
                "images": encoded_images,
                "disease_details": ["common_names", "url", "description", "treatment"]
            },
            timeout=10
        )
        disease_resp.raise_for_status()
        disease_data = disease_resp.json()
        if not disease_data.get('health_assessment'):
            error_msg = "Disease assessment failed."
            return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 400

        if disease_data['health_assessment']['is_healthy']:
            disease_name = "Healthy"
            remedy = f"""
## Plant Health Status
Your plant appears to be in good health. Here are some care recommendations to maintain its vitality:

### Watering Guidelines
- Maintain consistent watering schedule
- Avoid waterlogging and overwatering
- Adjust watering frequency based on {user_prefs['region'] or 'local'} climate conditions

### Environmental Care
- Ensure adequate sunlight as per plant requirements
- Provide proper ventilation
- Monitor temperature changes

### Maintenance
- Regular pruning: Remove dead or yellowing leaves
- Soil health: Use organic compost periodically
- pH monitoring: Check soil pH levels monthly

### Regional Considerations
For {user_prefs['region'] or 'India'}, consider seasonal variations like monsoon and dry periods when planning your care routine.
            """
            disease_confidence = disease_data['health_assessment']['is_healthy_probability']
            selected_model = 'none'
        else:
            issue = disease_data['health_assessment']['diseases'][0]
            disease_name = issue['name']
            disease_confidence = issue['probability']
            disease_description = issue.get('disease_details', {}).get('description', 'No description available')
            treatment_info = issue.get('disease_details', {}).get('treatment', {})
            biological = treatment_info.get('biological', ['No biological treatment available'])[0]
            chemical = treatment_info.get('chemical', ['No chemical treatment available'])[0]
            remedy = f"""
## Treatment Options
### Organic Treatment
{biological}

### Chemical Treatment
{chemical}

### Important Notes
- Always follow safety guidelines when applying treatments
- Consider environmental impact of chemical treatments
- Monitor plant response after treatment
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
            remedy = remedy_cache[cache_key_remedy]
            selected_model = 'cached'
        else:
            enhanced_remedy, selected_model = get_region_specific_remedy(
                plant_name, scientific_name, disease_name, user_prefs['region'], user_prefs['language'], remedy, disease_description
            )
            if enhanced_remedy:
                remedy = enhanced_remedy
                remedy_cache[cache_key_remedy] = remedy
            else:
                remedy += f"\n\n## Note\nFor {user_prefs['region'] or 'India'}, consult local agricultural extension services for region-specific advice."
                selected_model = 'none'

        formatted_remedy = format_text_for_display(remedy)

        return jsonify({
            "plant": translate_text(plant_name, "en", user_prefs['language']),
            "scientific_name": translate_text(scientific_name, "en", user_prefs['language']),
            "plant_confidence": f"{plant_confidence:.2%}",
            "disease": translate_text(disease_name, "en", user_prefs['language']),
            "disease_confidence": f"{disease_confidence:.2%}",
            "remedy": formatted_remedy,
            "model": selected_model,
            "typing_effect": True
        })
    except (IOError, requests.RequestException) as e:
        error_msg = f"Failed to process image or connect to API: {str(e)}"
        return jsonify({"error": translate_text(error_msg, "en", user_prefs['language']), "typing_effect": True}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
