# AgriBot: Your Smart Agricultural Assistant

AgriBot is an intelligent, web-based agricultural assistant designed to help farmers and gardeners in India. It provides expert advice on crop management, identifies plant diseases through image analysis, and offers localized, actionable remedies. The interface supports multiple Indian languages, making it accessible to a diverse user base.

---

## ✨ Features

* **🤖 Intelligent Chat:** Ask questions in natural language about crop selection, weather, fertilizers, and farming techniques.
* **📸 Plant Disease Detection:** Upload up to three images of a plant, and AgriBot will identify the plant and diagnose any potential diseases.
* **🌍 Region-Specific Advice:** Get agricultural advice tailored to your specific state or region in India.
* **🗣️ Multi-Language Support:** Interact with AgriBot in English or one of several Indian languages, including Hindi, Tamil, Bengali, and more.
* **⚡ Real-time Responses:** The chat interface provides streaming responses, making the interaction feel fast and responsive.
* **🌗 Light & Dark Mode:** A sleek, modern interface with toggleable light and dark themes for user comfort.
* **📱 Fully Responsive:** The user interface is designed to work seamlessly on desktops, tablets, and mobile devices.
* **🛑 Stoppable Responses:** Users can stop the AI from generating a response mid-stream.

---

## 🛠️ Tech Stack

### Frontend

* **HTML5 & CSS3:** For the core structure and styling of the application.
* **Tailwind CSS:** A utility-first CSS framework for rapid UI development.
* **Vanilla JavaScript:** For all client-side logic, including API calls, DOM manipulation, and handling user interactions.
* **Marked.js:** A library to parse and render Markdown responses from the AI.

### Backend

* **Python 3:** The primary language for the server-side logic.
* **Flask:** A lightweight web framework for building the backend API.
* **OpenRouter AI:** Powers the natural language understanding and text generation using the high-speed `google/gemini-flash-1.5` model.
* **Plant.id API:** Used for accurate plant identification and health assessment from user-uploaded images.
* **Pillow (PIL):** For image processing and optimization on the backend.

---

## 🚀 Setup and Installation

Follow these steps to get a local copy of AgriBot up and running.

### 1. Prerequisites

* Python 3.8 or higher
* `pip` (Python package installer)

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/agribot.git](https://github.com/your-username/agribot.git)
cd agribot
```

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
Flask
python-dotenv
requests
Pillow
markdown
bleach
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a file named `.env` in the root of your project directory. This file will store your secret API keys.

```env
# Get your key from [https://openrouter.ai/](https://openrouter.ai/)
OPENROUTER_API_KEY="your_openrouter_api_key"

# Get your key from [https://plant.id/](https://plant.id/)
PLANT_ID_API_KEY="your_plant_id_api_key"

# Optional: Change the default AI model
# OPENROUTER_MODEL="anthropic/claude-3-haiku"
```

### 6. Run the Application

Navigate into the `api` directory and start the Flask development server:

```bash
cd api
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000` to see AgriBot in action.

---

##  usage How to Use AgriBot

1.  **Set Your Preferences:** Use the dropdown menus to select your region (state) in India and your preferred language.
2.  **Ask a Question:** Type any agricultural question into the chat input at the bottom and press Enter or click the send button.
3.  **Upload an Image:** Click the paperclip icon to select up to three images of a plant you want to analyze. The bot will automatically identify the plant and check for diseases.
4.  **Stop a Response:** If a response is taking too long or you want to ask something else, click the red "Stop" button that appears while the bot is "typing".

---

## 📂 Project Structure

```
.
├── api/
│   └── app.py            # The main Flask backend server
├── templates/
│   └── index.html        # The single-page frontend for the application
├── .env                  # Stores API keys and environment variables
├── requirements.txt      # Python package dependencies
└── README.md             # This file
```

---

## 💡 Future Improvements

* **User Accounts & History:** Implement user authentication to save conversation history.
* **Database Integration:** Store user preferences and chat history in a database like SQLite or PostgreSQL.
* **Expanded Knowledge Base:** Integrate more agricultural data sources, such as soil type maps and detailed crop calendars.
* **Real-time Weather API:** Provide up-to-the-minute weather forecasts and alerts.
