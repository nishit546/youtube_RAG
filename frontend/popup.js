const askBtn = document.getElementById("askBtn");
const videoInput = document.getElementById("videoId");
const questionInput = document.getElementById("question");
const chatDiv = document.getElementById("chat");
const statusDiv = document.getElementById("status");
const errorDiv = document.getElementById("error");

function addMessage(text, type) {
  const msg = document.createElement("div");
  msg.className = `msg ${type}`;
  msg.innerText = text;
  chatDiv.appendChild(msg);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

function setLoading(isLoading) {
  askBtn.disabled = isLoading;
  statusDiv.style.display = isLoading ? "block" : "none";
  statusDiv.innerText = isLoading ? "Thinking..." : "";
}

askBtn.addEventListener("click", async () => {
  const video_id = videoInput.value.trim();
  const question = questionInput.value.trim();

  errorDiv.innerText = "";

  if (!video_id) {
    errorDiv.innerText = "Please enter a YouTube video ID.";
    return;
  }

  if (!question) {
    errorDiv.innerText = "Please enter a question.";
    return;
  }

  addMessage(question, "user");
  setLoading(true);

  try {
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id, question })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Something went wrong");
    }

    addMessage(data.answer, "bot");
    questionInput.value = "";
  } catch (err) {
    console.error("Error:", err);
    
    // More descriptive error messages
    let errorMessage = "Failed to connect to server.";
    
    if (err.message.includes("Failed to fetch") || err.name === "TypeError") {
      errorMessage = "⚠️ Server not running! Please start the backend server on port 8000.";
    } else if (err.message.includes("Transcript not available")) {
      errorMessage = "❌ No transcript available for this video. Try another video.";
    } else if (err.message) {
      errorMessage = err.message;
    }
    
    errorDiv.innerText = errorMessage;
    addMessage(`Error: ${errorMessage}`, "bot");
  } finally {
    setLoading(false);
  }
});

// Optional: Press Enter to ask
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    askBtn.click();
  }
});
