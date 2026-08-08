import React, { useEffect, useState } from "react";
import "./styles.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";


function PredictionResult({ result }) {
  if (!result?.prediction || !result?.guidance_rw) {
    return null;
  }

  const prediction = result.prediction;
  const guidance = result.guidance_rw;

  return (
    <section className="card result" aria-live="polite">
      <p className="eyebrow">ICYABONETSE</p>

      <h2>{prediction.label_en}</h2>

      <p className="kinyarwanda-name">
        {prediction.label_rw}
      </p>

      <p className="result-message">
        {guidance.message}
      </p>

      <div className="next-step">
        <h3>Icyo wakora</h3>
        <p>{guidance.next_step}</p>
      </div>

      <div className="warning">
        <h3>Icyitonderwa</h3>
        <p>{guidance.warning}</p>
      </div>
    </section>
  );
}


export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
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

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setStatus("");
  }


  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setStatus("Hitamo ifoto mbere.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setStatus("Tegereza, ifoto iri gusuzumwa...");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(
          payload.detail || "Gusuzuma ifoto ntibyakunze."
        );
      }

      if (!payload.prediction || !payload.guidance_rw) {
        console.log("Unexpected API response:", payload);

        throw new Error(
          "Igisubizo cya seriveri nticyumvikanye."
        );
      }

      // Save the complete response from FastAPI.
      setResult(payload);
      setStatus("");

    } catch (error) {
      console.error(error);

      setStatus(
        error.message ||
          "Ntibyashobotse kugera kuri seriveri."
      );

    } finally {
      setIsLoading(false);
    }
  }


  return (
    <main className="shell">
      <p className="eyebrow">
        KIGALI E-WASTE CLASSIFIER
      </p>

      <h1>
        Identify an electronic item before sorting it.
      </h1>

      <p className="intro">
        Upload one clear photo to identify the electronic
        waste item.
      </p>

      <form className="card" onSubmit={handleSubmit}>
        <label
          className="dropzone"
          htmlFor="image-input"
        >
          {preview ? (
            <img
              className="preview"
              src={preview}
              alt="Selected electronic waste item"
            />
          ) : (
            <>
              <span className="dropzone-title">
                Hitamo ifoto
              </span>

              <span>JPG, PNG cyangwa WEBP</span>
            </>
          )}
        </label>

        <input
          id="image-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />

        {file && (
          <p className="file-name">
            {file.name}
          </p>
        )}

        <button
          type="submit"
          disabled={!file || isLoading}
        >
          {isLoading
            ? "Ifoto iri gusuzumwa..."
            : "Suzuma igikoresho"}
        </button>

        {status && (
          <p className="status" role="status">
            {status}
          </p>
        )}
      </form>

      <PredictionResult result={result} />
    </main>
  );
}