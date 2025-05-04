const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const typingIndicator = document.getElementById("typing-indicator");

function appendMessage(sender, text, emotion = "") {
  const div = document.createElement("div");
  div.className = sender;
  const emoji = emotion ? ` ${getEmoji(emotion)}` : "";
  div.textContent = `${sender === "user" ? "You" : "Dr. Mind"}: ${text}${emoji}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function getEmoji(emotion) {
  const map = {
    joy: "😊", sadness: "😢", anger: "😠",
    fear: "😱", love: "❤️", surprise: "😲",
  };
  return map[emotion.toLowerCase()] || "";
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;
  appendMessage("user", message);
  userInput.value = "";

  typingIndicator.classList.remove("hidden");

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const data = await res.json();
  typingIndicator.classList.add("hidden");

  if (data.response) {
    appendMessage("bot", data.response, data.detected_emotion);
  } else {
    appendMessage("bot", "⚠️ " + data.error);
  }
}

function clearChat() {
  chatBox.innerHTML = "";
}
