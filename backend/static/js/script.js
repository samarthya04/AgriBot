let currentRegion = "";
let currentLanguage = "english";
let uploadedImage = null;
let imageFile = null;
let isProcessing = false;

document.addEventListener("DOMContentLoaded", () => {
    const regionSelect = document.getElementById("region");
    const languageSelect = document.getElementById("language");
    const messageInput = document.getElementById("messageInput");
    const imageInput = document.getElementById("imageInput");
    const sendBtn = document.getElementById("sendBtn");

    regionSelect.addEventListener("change", (e) => {
        currentRegion = e.target.value;
        if (currentRegion) {
            addMessage("system", `Region set to: ${e.target.options[e.target.selectedIndex].text}`);
        }
    });

    languageSelect.addEventListener("change", (e) => {
        currentLanguage = e.target.value;
        addMessage("system", `Language set to: ${e.target.options[e.target.selectedIndex].text}`);
    });

    messageInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !isProcessing) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 100) + "px";
    });

    imageInput.addEventListener("change", handleImageUpload);

    checkAPIStatus();
});

function addMessage(type, content, image = null) {
    const messagesContainer = document.getElementById("messages");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}`;
    messageDiv.innerHTML = image ? `${content}<br><img src="${image}" class="image-preview" alt="Uploaded plant image">` : content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showTyping() {
    const messagesContainer = document.getElementById("messages");
    const typingDiv = document.createElement("div");
    typingDiv.className = "message bot typing";
    typingDiv.id = "typing-indicator";
    typingDiv.innerHTML = `
        <span>🤖 AI is analyzing...</span>
        <div class="typing-dots"><span></span><span></span><span></span></div>
    `;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTyping() {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) typingIndicator.remove();
}

async function handleImageUpload(event) {
    if (isProcessing) return;
    const file = event.target.files[0];
    if (!file) return;

    if (file.size > 16 * 1024 * 1024) {
        addMessage("error", "❌ File too large. Please select an image under 16MB.");
        return;
    }

    if (!file.type.startsWith("image/")) {
        addMessage("error", "❌ Please select a valid image file.");
        return;
    }

    imageFile = file;
    const reader = new FileReader();
    reader.onload = async (e) => {
        uploadedImage = e.target.result;
        addMessage("user", "📸 Image uploaded for AI analysis", uploadedImage);
        await analyzeImage(file);
    };
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    showTyping();
    isProcessing = true;
    document.getElementById("sendBtn").disabled = true;

    try {
        const formData = new FormData();
        formData.append("image", file);
        const uploadResponse = await fetch("/api/upload-image", {
            method: "POST",
            body: formData,
        }).then((res) => res.json());

        if (!uploadResponse.success) {
            addMessage("error", `❌ Upload failed: ${uploadResponse.error}`);
            return;
        }

        const { image_data, image_hash } = uploadResponse;
        const region = currentRegion;

        const [plantResult, speciesResult] = await Promise.allSettled([
            fetch("/api/identify-plant", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: image_data, region }),
            }).then((res) => res.json()),
            fetch("/api/identify-species", {
                method: "POST",
                body: formData,
            }).then((res) => res.json()),
        ]);

        let analysisResult;
        if (plantResult.status === "fulfilled" && plantResult.value.success) {
            analysisResult = plantResult.value;
        } else if (speciesResult.status === "fulfilled" && speciesResult.value.success) {
            analysisResult = speciesResult.value;
        } else {
            addMessage("error", "⚠️ API services unavailable. Using local analysis...");
            const localResult = await fetch("/api/local-analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ region, image_hash }),
            }).then((res) => res.json());
            if (localResult.success) {
                displayLocalAnalysis(localResult.data);
            } else {
                addMessage("error", "❌ Local analysis failed.");
            }
            return;
        }

        if (analysisResult.data.type === "health_assessment") {
            displayHealthResult(analysisResult.data);
        } else {
            displaySpeciesResult(analysisResult.data);
        }
    } catch (error) {
        addMessage("error", "❌ Error analyzing image. Please try again.");
        console.error("Analysis error:", error);
    } finally {
        hideTyping();
        isProcessing = false;
        document.getElementById("sendBtn").disabled = false;
        uploadedImage = null;
        imageFile = null;
    }
}

function displayHealthResult(data) {
    let response = `🔬 <strong>AI Plant Health Analysis</strong><br><br>`;
    response += `<div class="analysis-result">`;
    response += `<strong>🌿 Plant:</strong> ${data.plant_name}<br>`;
    response += `<strong>🎯 Confidence:</strong><br>`;
    response += `<div class="confidence-bar"><div class="confidence-fill" style="width: ${Math.round(data.probability * 100)}%">${Math.round(data.probability * 100)}%</div></div>`;

    if (data.diseases && data.diseases.length > 0) {
        response += `<br><strong>🦠 Detected Issues:</strong><br>`;
        data.diseases.forEach((disease) => {
            response += `• <strong>${disease.name}</strong> (${Math.round(disease.probability * 100)}% confidence)<br>`;
            if (disease.disease_details) response += `  ${disease.disease_details.description}<br>`;
        });
        response += `<br><strong>💊 Treatment Recommendations:</strong><br>`;
        response += `• Apply appropriate fungicide<br>• Improve air circulation<br>• Remove affected leaves<br>• Monitor plant closely<br>`;
    } else {
        response += `<br>✅ <strong>Plant appears healthy!</strong><br>`;
    }
    response += `</div>`;
    addMessage("bot", response);
}

function displaySpeciesResult(data) {
    let response = `🔍 <strong>Plant Identification Results</strong><br><br>`;
    response += `<div class="analysis-result">`;
    response += `<strong>🌱 Species:</strong> ${data.species.scientificNameWithoutAuthor}<br>`;
    response += `<strong>🏷️ Common Name:</strong> ${data.species.commonNames ? data.species.commonNames[0] : "Not available"}<br>`;
    response += `<strong>👨‍🔬 Scientific Name:</strong> ${data.species.scientificNameAuthorship}<br>`;
    response += `<strong>🎯 Confidence:</strong><br>`;
    response += `<div class="confidence-bar"><div class="confidence-fill" style="width: ${Math.round(data.score * 100)}%">${Math.round(data.score * 100)}%</div></div>`;
    response += `</div>`;
    addMessage("bot", response);
}

function displayLocalAnalysis(data) {
    let response = `🔍 <strong>Local Analysis Results</strong><br><br>`;
    response += `<div class="analysis-result">`;
    response += `<strong>📋 Condition:</strong> ${data.condition}<br>`;
    response += `<strong>🎯 Confidence:</strong><br>`;
    response += `<div class="confidence-bar"><div class="confidence-fill" style="width: ${data.confidence}%">${data.confidence}%</div></div>`;
    response += `<br><strong>👁️ Symptoms:</strong> ${data.symptoms}<br>`;
    response += `<strong>💊 Treatment:</strong> ${data.treatment}<br>`;
    response += `<strong>🛡️ Prevention:</strong> ${data.prevention}`;
    response += `</div>`;
    addMessage("bot", response);
}

async function sendQuickMessage(message) {
    if (isProcessing) return;
    document.getElementById("messageInput").value = message;
    sendMessage();
}

async function sendMessage() {
    if (isProcessing) return;
    const messageInput = document.getElementById("messageInput");
    let message = messageInput.value.trim();

    if (!message && !uploadedImage) return;

    if (!currentRegion && message) {
        addMessage("system", "⚠️ Please select your region first.");
        return;
    }

    isProcessing = true;
    document.getElementById("sendBtn").disabled = true;

    if (message) {
        addMessage("user", message);
        messageInput.value = "";
        messageInput.style.height = "auto";
    }

    showTyping();
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, region: currentRegion, language: currentLanguage }),
        }).then((res) => res.json());

        if (response.success) {
            addMessage("bot", response.response);
        } else {
            addMessage("error", `❌ Error: ${response.error || "Failed to process request."}`);
        }
    } catch (error) {
        addMessage("error", "❌ Error processing request. Please try again.");
        console.error("Chat error:", error);
    } finally {
        hideTyping();
        isProcessing = false;
        document.getElementById("sendBtn").disabled = false;
    }
}

async function checkAPIStatus() {
    try {
        const response = await fetch("/api/check-status").then((res) => res.json());
        response.forEach(({ id, name, status }) => {
            const element = document.getElementById(id);
            element.textContent = `🔍 ${name}: ${status}`;
            element.classList.remove("status-online", "status-offline");
            element.classList.add(status === "Active" ? "status-online" : "status-offline");
        });
    } catch (error) {
        console.error("API status check failed:", error);
    }
}