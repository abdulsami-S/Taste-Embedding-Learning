const API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "";

// Chat state
let chatHistory = [];

// DOM Elements
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const submitBtn = document.getElementById("submit-btn");
const loadingIndicator = document.getElementById("loading-indicator");
const chatContainer = document.getElementById("chat-container");
const examplePrompts = document.getElementById("example-prompts");
const newChatBtn = document.getElementById("new-chat-btn");
const warningMsg = document.getElementById("warning-msg");

// Similar Search Elements
const similarForm = document.getElementById("similar-form");
const similarName = document.getElementById("similar-name");
const similarIngredients = document.getElementById("similar-ingredients");
const similarResults = document.getElementById("similar-results");
const similarSubmitBtn = document.getElementById("similar-submit-btn");

// Toggles the loading state for the chat
function setLoading(isLoading) {
  if (isLoading) {
    loadingIndicator.classList.remove("hidden");
    submitBtn.disabled = true;
    questionInput.disabled = true;
  } else {
    loadingIndicator.classList.add("hidden");
    submitBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

// Resets the chat history and UI
function resetChat() {
  chatHistory = [];
  chatMessages.innerHTML = "";
  examplePrompts.style.display = "flex";
  warningMsg.classList.add("hidden");
}

// Renders the recipe cards below an assistant message
function renderRecipeCards(results, container) {
  if (!results || results.length === 0) return;
  
  const cardsContainer = document.createElement("div");
  cardsContainer.style.marginTop = "0.75rem";
  cardsContainer.style.display = "flex";
  cardsContainer.style.flexDirection = "column";
  cardsContainer.style.gap = "0.5rem";

  results.forEach(recipe => {
    const card = document.createElement("div");
    card.className = "history-item";
    card.style.textAlign = "left";

    // Title and Match Score
    const titleRow = document.createElement("div");
    titleRow.className = "history-question";
    
    const titleSpan = document.createElement("span");
    titleSpan.textContent = recipe.name;
    
    const scoreSpan = document.createElement("span");
    scoreSpan.style.float = "right";
    scoreSpan.style.color = "var(--forest)";
    scoreSpan.textContent = Math.round(recipe.score * 100) + "% Match";
    
    titleRow.appendChild(titleSpan);
    titleRow.appendChild(scoreSpan);
    card.appendChild(titleRow);

    // Ingredients List
    const ingPreview = document.createElement("div");
    ingPreview.className = "history-preview";
    
    const ingredients = recipe.ingredients || [];
    let showAll = false;
    const ingText = document.createElement("span");
    
    const updateIngText = () => {
      if (ingredients.length > 8 && !showAll) {
        ingText.textContent = "Ingredients: " + ingredients.slice(0, 8).join(", ") + "... ";
      } else {
        ingText.textContent = "Ingredients: " + ingredients.join(", ") + " ";
      }
    };
    updateIngText();
    ingPreview.appendChild(ingText);

    // Toggle button for long ingredient lists
    if (ingredients.length > 8) {
      const toggleBtn = document.createElement("button");
      toggleBtn.textContent = "Show all";
      toggleBtn.style.background = "none";
      toggleBtn.style.border = "none";
      toggleBtn.style.color = "var(--forest)";
      toggleBtn.style.cursor = "pointer";
      toggleBtn.style.textDecoration = "underline";
      toggleBtn.style.fontSize = "0.8rem";
      toggleBtn.addEventListener("click", () => {
        showAll = !showAll;
        toggleBtn.textContent = showAll ? "Show less" : "Show all";
        updateIngText();
      });
      ingPreview.appendChild(toggleBtn);
    }
    
    card.appendChild(ingPreview);

    // Tags
    if (recipe.search_terms && recipe.search_terms.length > 0) {
      const tagsContainer = document.createElement("div");
      tagsContainer.className = "answer-meta";
      tagsContainer.style.marginTop = "0.5rem";
      tagsContainer.style.marginBottom = "0";
      
      const tags = recipe.search_terms.slice(0, 4);
      tags.forEach(tag => {
        const chip = document.createElement("span");
        chip.className = "meta-chip";
        chip.style.fontSize = "0.75rem";
        chip.style.padding = "0.2rem 0.5rem";
        chip.textContent = tag;
        tagsContainer.appendChild(chip);
      });
      card.appendChild(tagsContainer);
    }

    cardsContainer.appendChild(card);
  });
  
  container.appendChild(cardsContainer);
}

// Renders a single message bubble (user or assistant)
function renderMessage(role, content, results = null) {
  const wrapper = document.createElement("div");
  wrapper.style.marginBottom = "1rem";
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";

  const bubble = document.createElement("div");
  bubble.className = "answer-text";
  bubble.style.display = "inline-block";
  bubble.style.whiteSpace = "pre-wrap";
  
  if (role === "user") {
    wrapper.style.alignItems = "flex-end";
    bubble.style.background = "var(--sage-pale)";
    bubble.style.borderLeft = "none";
    bubble.style.borderRight = "4px solid var(--forest)";
    bubble.style.borderRadius = "var(--radius-md) 0 var(--radius-md) var(--radius-md)";
    bubble.style.textAlign = "right";
  } else {
    wrapper.style.alignItems = "flex-start";
    bubble.style.borderRadius = "0 var(--radius-md) var(--radius-md) var(--radius-md)";
    bubble.style.textAlign = "left";
  }

  bubble.textContent = content;
  wrapper.appendChild(bubble);

  // Append recipe cards if provided by assistant
  if (results) {
    const cardsWrapper = document.createElement("div");
    cardsWrapper.style.width = "100%";
    renderRecipeCards(results, cardsWrapper);
    wrapper.appendChild(cardsWrapper);
  }

  chatMessages.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Sends the user message to the API and handles the response
async function sendMessage(text) {
  if (!text.trim()) return;
  
  examplePrompts.style.display = "none";
  warningMsg.classList.add("hidden");
  
  renderMessage("user", text);
  chatHistory.push({ role: "user", content: text });
  
  setLoading(true);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  
  try {
    const payload = {
      messages: chatHistory.slice(-10),
      k: 5
    };
    
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    const data = await response.json();
    chatHistory.push({ role: "assistant", content: data.reply });
    renderMessage("assistant", data.reply, data.results);
    
  } catch (err) {
    console.error(err);
    renderMessage("assistant", "I couldn't reach the server. Is it running on port 8000?");
  } finally {
    setLoading(false);
  }
}

// Event Listeners for Chat
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = questionInput.value;
  questionInput.value = "";
  sendMessage(text);
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});

document.querySelectorAll(".example-prompt").forEach(btn => {
  btn.addEventListener("click", () => {
    questionInput.value = btn.textContent;
    chatForm.dispatchEvent(new Event("submit"));
  });
});

newChatBtn.addEventListener("click", resetChat);

// Event Listeners for Similar Search
similarForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = similarName.value.trim();
  const ings = similarIngredients.value.split(",").map(i => i.trim()).filter(i => i);
  
  if (!name || ings.length === 0) return;
  
  similarSubmitBtn.disabled = true;
  similarSubmitBtn.textContent = "Searching...";
  similarResults.innerHTML = "";
  
  try {
    const response = await fetch(`${API_BASE}/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, ingredients: ings, k: 5 })
    });
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    const data = await response.json();
    renderRecipeCards(data.results, similarResults);
    
  } catch (err) {
    console.error(err);
    similarResults.textContent = "Error fetching similar recipes. Is the server running?";
    similarResults.style.color = "var(--danger)";
  } finally {
    similarSubmitBtn.disabled = false;
    similarSubmitBtn.textContent = "Search";
  }
});
