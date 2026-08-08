import React, { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function PredictionResult({ predictions }) {
  if (!predictions?.length) return null;

  const [top, ...alternatives] = predictions;

  return (
    <section className="card result">
      <p className="eyebrow">RESULT</p>

      <h2>{top.label}</h2>

      <p className="confidence">
        {Math.round(top.confidence * 100)}% confidence
      </p>

      {alternatives.length > 0 && (
        <div className="alternatives">
          <p>Other possibilities</p>

          {alternatives.map((prediction) => (
            <div className="alternative" key={prediction.label}>
              <span>{prediction.label}</span>
              <span>
                {Math.round(prediction.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="safety">
        This is a device-category suggestion, not a hazard assessment.
        If the item is damaged, leaking, hot, a battery, or a CRT, follow
        your organization&apos;s safety procedure.
      </p>
    </section>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [status, setStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setPredictions(null);
    setStatus("");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setStatus("Classifying...");
    setPredictions(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed");
      }

      setPredictions(payload.predictions);
      setStatus("Done.");
    } catch (error) {
      setStatus(
        error.message || "Could not reach the prediction service."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: "700px",
        margin: "0 auto",
        padding: "40px 20px",
        fontFamily: "Arial, sans-serif",
        color: "#17342a",
      }}
    >
      <p>KIGALI E-WASTE CLASSIFIER</p>

      <h1>Identify an electronic item before sorting it.</h1>

      <p>
        Upload one clear photo to classify the electronic waste item.
      </p>

      <form onSubmit={handleSubmit}>
        <label
          htmlFor="image-input"
          style={{
            display: "block",
            padding: "50px 20px",
            border: "2px dashed #5b9b72",
            borderRadius: "12px",
            textAlign: "center",
            cursor: "pointer",
          }}
        >
          {preview ? (
            <img
              src={preview}
              alt="Selected e-waste item"
              style={{
                maxWidth: "100%",
                maxHeight: "250px",
              }}
            />
          ) : (
            <>
              <strong>Choose an image</strong>
              <br />
              <span>JPG, PNG, or WEBP</span>
            </>
          )}
        </label>

        <input
          id="image-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        {file && <p>{file.name}</p>}

        <button
          type="submit"
          disabled={!file || isLoading}
          style={{
            width: "100%",
            marginTop: "20px",
            padding: "15px",
            border: "none",
            borderRadius: "8px",
            background: "#1d7049",
            color: "white",
            cursor: "pointer",
          }}
        >
          {isLoading ? "Classifying..." : "Classify item"}
        </button>

        <p>{status}</p>
      </form>

      <PredictionResult predictions={predictions} />
    </main>
  );
}