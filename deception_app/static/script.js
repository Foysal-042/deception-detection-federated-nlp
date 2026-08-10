/*
  script.js
  ---------
  Handles the "Analyze" button: reads the textarea + language dropdown,
  calls the Flask /predict endpoint, and fills in the Prediction Result card.
*/

document.addEventListener("DOMContentLoaded", () => {
  const textInput = document.getElementById("input-text");
  const languageSelect = document.getElementById("language-select");
  const analyzeBtn = document.getElementById("analyze-btn");
  const errorMessage = document.getElementById("error-message");

  const resultCard = document.getElementById("result-card");
  const resultPrediction = document.getElementById("result-prediction");
  const resultConfidence = document.getElementById("result-confidence");
  const resultModel = document.getElementById("result-model");

  analyzeBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    const language = languageSelect.value;

    hideError();

    if (!text) {
      showError("Please enter some text to analyze.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "Something went wrong while analyzing the text.");
        return;
      }

      renderResult(data);
    } catch (err) {
      showError("Could not reach the server. Please check that the Flask app is running.");
    } finally {
      setLoading(false);
    }
  });

  function renderResult(data) {
    resultPrediction.textContent = data.prediction;
    resultPrediction.classList.remove("truthful", "deceptive");
    resultPrediction.classList.add(
      data.prediction === "Truthful" ? "truthful" : "deceptive"
    );

    resultConfidence.textContent = `${data.confidence}%`;
    resultModel.textContent = data.model || "BanglishBERT (Federated)";

    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    analyzeBtn.textContent = isLoading ? "Analyzing..." : "Analyze";
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  }

  function hideError() {
    errorMessage.hidden = true;
    errorMessage.textContent = "";
  }
});
