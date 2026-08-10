import React, { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const COPY = {
  rw: {
    heading: "Menya igikoresho mbere yo kugishyira mu byiciro.",
    intro: "Fata cyangwa ohereza ifoto isobanutse. Tuzakwereka ubwoko bw'igikoresho n'icyo wakora.",
    takePhoto: "Fata ifoto",
    uploadPhoto: "Ohereza ifoto",
    cameraHint: "Koresha kamera ya telefoni cyangwa mudasobwa",
    uploadHint: "Hitamo ifoto iri muri telefoni cyangwa mudasobwa",
    previewReady: "Ifoto yiteguye gusuzumwa",
    changePhoto: "Hindura ifoto",
    classify: "Suzuma igikoresho",
    classifying: "Ifoto iri gusuzumwa...",
    selectFirst: "Banza ufate cyangwa wohereze ifoto.",
    requestFailed: "Gusuzuma ifoto ntibyakunze.",
    connectionFailed: "Ntibyashobotse kugera kuri seriveri.",
    invalidResponse: "Seriveri yatanze igisubizo kitumvikana.",
    result: "Icyabonetse",
    nextStep: "Icyo wakora",
    warning: "Icyitonderwa",
    cameraTitle: "Fata ifoto",
    cameraHelp: "Shyira igikoresho cyose hagati mu ishusho.",
    cameraStarting: "Kamera iri gufunguka...",
    cameraError: "Kamera ntiyafungutse. Emerera uru rubuga gukoresha kamera cyangwa wohereze ifoto isanzwe.",
    close: "Funga kamera",
    switchCamera: "Hindura kamera",
    capture: "Fata",
  },
  en: {
    heading: "Identify an electronic item before sorting it.",
    intro: "Take or upload one clear photo. We will show the item category and what to do next.",
    takePhoto: "Take a photo",
    uploadPhoto: "Upload image",
    cameraHint: "Use your phone or laptop camera",
    uploadHint: "Choose an image already on your device",
    previewReady: "Photo ready for classification",
    changePhoto: "Change photo",
    classify: "Classify item",
    classifying: "Classifying image...",
    selectFirst: "Take or upload an image first.",
    requestFailed: "Image classification failed.",
    connectionFailed: "Could not reach the prediction service.",
    invalidResponse: "The server returned an unexpected response.",
    result: "Result",
    nextStep: "What to do next",
    warning: "Safety note",
    cameraTitle: "Take a photo",
    cameraHelp: "Keep the entire item inside the frame.",
    cameraStarting: "Starting camera...",
    cameraError: "The camera could not start. Allow camera access or upload an existing image instead.",
    close: "Close camera",
    switchCamera: "Switch camera",
    capture: "Capture",
  },
};

function CameraIcon({ className = "h-6 w-6" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M14.5 5 13 3.5h-2L9.5 5H6a3 3 0 0 0-3 3v9a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3V8a3 3 0 0 0-3-3h-3.5Z" />
      <circle cx="12" cy="12.5" r="4" />
    </svg>
  );
}

function UploadIcon({ className = "h-6 w-6" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
    </svg>
  );
}

function SparkIcon({ className = "h-5 w-5" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3 10.7 8.1 7 11l3.7 2.9L12 19l1.3-5.1L17 11l-3.7-2.9L12 3Z" />
    </svg>
  );
}

function getEnglishGuidance(label) {
  if (label === "Unknown") {
    return {
      message: "This item could not be identified confidently.",
      next_step: "Do not put it in general waste. Ask a trained worker to inspect it.",
      warning: "Do not open or burn it, and do not touch it if it is hot or leaking.",
    };
  }
  if (label === "Battery") {
    return {
      message: "This item is a battery.",
      next_step: "Keep it separate and take it to an approved e-waste collection point.",
      warning: "Do not open, crush, puncture or burn it.",
    };
  }
  if (label === "Mobile") {
    return {
      message: "This item is a mobile phone.",
      next_step: "Keep it out of general waste and take it to an approved e-waste collection point.",
      warning: "Do not open or burn it because it may contain a battery.",
    };
  }
  return {
    message: `This item is classified as ${label}.`,
    next_step: "Keep it separate from general waste and take it to an approved e-waste collection point.",
    warning: "Do not burn, crush or dismantle the item.",
  };
}

function CameraModal({ language, onClose, onCapture }) {
  const text = COPY[language];
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function startCamera() {
      setReady(false);
      setError("");

      if (!navigator.mediaDevices?.getUserMedia) {
        setError(text.cameraError);
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 1280 },
          },
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setReady(true);
        }
      } catch (cameraError) {
        console.error(cameraError);
        setError(text.cameraError);
      }
    }

    startCamera();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [facingMode, text.cameraError]);

  function switchCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setFacingMode((current) => current === "environment" ? "user" : "environment");
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const photo = new File([blob], `ewaste-${Date.now()}.jpg`, { type: "image/jpeg" });
      onCapture(photo);
    }, "image/jpeg", 0.9);
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-950 text-white">
      <div className="mx-auto flex w-full max-w-lg items-center justify-between px-4 py-4">
        <div>
          <p className="text-lg font-bold">{text.cameraTitle}</p>
          <p className="text-sm text-white/65">{text.cameraHelp}</p>
        </div>
        <button type="button" onClick={onClose} aria-label={text.close} className="grid h-11 w-11 place-items-center rounded-full bg-white/10 text-2xl transition hover:bg-white/20">
          ×
        </button>
      </div>

      <div className="relative mx-auto flex w-full max-w-lg flex-1 items-center overflow-hidden bg-black">
        <video ref={videoRef} autoPlay muted playsInline className="h-full max-h-[70dvh] w-full object-cover" />

        {!ready && !error && (
          <div className="absolute inset-0 grid place-items-center bg-slate-950">
            <div className="text-center">
              <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-white/20 border-t-emerald-400" />
              <p className="text-sm text-white/70">{text.cameraStarting}</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 grid place-items-center bg-slate-950 px-8 text-center">
            <div>
              <CameraIcon className="mx-auto mb-4 h-12 w-12 text-amber-300" />
              <p className="leading-6 text-white/80">{error}</p>
              <button type="button" onClick={onClose} className="mt-6 rounded-full bg-white px-6 py-3 font-bold text-slate-900">
                {text.close}
              </button>
            </div>
          </div>
        )}

        {ready && <div className="pointer-events-none absolute inset-6 rounded-[2rem] border-2 border-white/70 shadow-[0_0_0_9999px_rgba(0,0,0,0.18)]" />}
      </div>

      <div className="mx-auto flex w-full max-w-lg items-center justify-around px-8 py-6">
        <button type="button" onClick={switchCamera} disabled={!ready} className="flex w-20 flex-col items-center gap-1 text-xs text-white/75 disabled:opacity-30">
          <span className="grid h-11 w-11 place-items-center rounded-full bg-white/10 text-xl">↻</span>
          {text.switchCamera}
        </button>

        <button type="button" onClick={capturePhoto} disabled={!ready} aria-label={text.capture} className="grid h-20 w-20 place-items-center rounded-full border-4 border-white bg-white/20 shadow-lg transition active:scale-95 disabled:opacity-30">
          <span className="h-14 w-14 rounded-full bg-white" />
        </button>

        <div className="w-20" />
      </div>
    </div>
  );
}

function PredictionResult({ result, language }) {
  if (!result?.prediction || !result?.guidance_rw) return null;

  const prediction = result.prediction;
  const text = COPY[language];
  const guidance = language === "rw" ? result.guidance_rw : getEnglishGuidance(prediction.label_en);
  const primaryName = language === "rw" ? prediction.label_rw : prediction.label_en;
  const secondaryName = language === "rw" ? prediction.label_en : prediction.label_rw;

  return (
    <section className="mt-5 overflow-hidden rounded-[2rem] border border-emerald-100 bg-white shadow-[0_20px_60px_-30px_rgba(6,78,59,0.45)]" aria-live="polite">
      <div className="bg-gradient-to-r from-emerald-600 to-teal-500 px-5 py-4 text-white">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-emerald-50">
          <SparkIcon /> {text.result}
        </div>
        <h2 className="mt-2 text-3xl font-black tracking-tight">{primaryName}</h2>
        <p className="mt-1 text-sm font-semibold text-white/75">{secondaryName}</p>
      </div>

      <div className="space-y-3 p-5">
        <p className="text-[15px] leading-6 text-slate-700">{guidance.message}</p>
        <div className="rounded-2xl bg-emerald-50 p-4">
          <p className="text-sm font-extrabold text-emerald-900">{text.nextStep}</p>
          <p className="mt-1 text-sm leading-6 text-emerald-900/75">{guidance.next_step}</p>
        </div>
        <div className="rounded-2xl bg-amber-50 p-4">
          <p className="text-sm font-extrabold text-amber-900">{text.warning}</p>
          <p className="mt-1 text-sm leading-6 text-amber-900/75">{guidance.warning}</p>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [language, setLanguage] = useState("rw");
  const [cameraOpen, setCameraOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const text = COPY[language];

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function selectImage(selectedFile) {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setStatus("");
  }

  function handleUpload(event) {
    selectImage(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleCameraCapture(photo) {
    selectImage(photo);
    setCameraOpen(false);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setStatus(text.selectFirst);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setIsLoading(true);
    setStatus(text.classifying);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || text.requestFailed);
      if (!payload.prediction || !payload.guidance_rw) throw new Error(text.invalidResponse);
      setResult(payload);
      setStatus("");
    } catch (error) {
      setStatus(error.message || text.connectionFailed);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-dvh bg-gradient-to-b from-emerald-50 via-white to-lime-50 text-slate-900">
      <main className="mx-auto w-full max-w-md px-4 pb-12 pt-5 sm:pt-8">
        <header className="mb-7">
          <div className="mb-7 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-600 text-lg font-black text-white shadow-lg shadow-emerald-600/20">E</div>
              <div>
                <p className="text-sm font-black tracking-tight text-emerald-950">Kigali E-Waste</p>
                <p className="text-[11px] font-semibold text-emerald-700/70">Smart sorting assistant</p>
              </div>
            </div>

            <div className="flex rounded-full border border-emerald-200 bg-white p-1 text-xs font-bold shadow-sm">
              <button type="button" onClick={() => setLanguage("rw")} className={`rounded-full px-3 py-2 transition ${language === "rw" ? "bg-emerald-600 text-white shadow-sm" : "text-slate-500"}`}>RW</button>
              <button type="button" onClick={() => setLanguage("en")} className={`rounded-full px-3 py-2 transition ${language === "en" ? "bg-emerald-600 text-white shadow-sm" : "text-slate-500"}`}>EN</button>
            </div>
          </div>

          <div className="inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1.5 text-xs font-extrabold text-emerald-800">
            <SparkIcon className="h-4 w-4" /> AI-powered sorting
          </div>
          <h1 className="mt-4 text-[2.35rem] font-black leading-[1.03] tracking-[-0.045em] text-emerald-950">{text.heading}</h1>
          <p className="mt-4 text-[15px] leading-6 text-slate-600">{text.intro}</p>
        </header>

        <form onSubmit={handleSubmit} className="rounded-[2rem] border border-white bg-white/90 p-4 shadow-[0_24px_70px_-36px_rgba(6,78,59,0.5)] backdrop-blur">
          {preview ? (
            <div>
              <div className="relative overflow-hidden rounded-[1.5rem] bg-slate-100">
                <img src={preview} alt="Selected electronic waste item" className="h-64 w-full object-contain" />
                <div className="absolute bottom-3 left-3 rounded-full bg-slate-950/75 px-3 py-1.5 text-xs font-bold text-white backdrop-blur">✓ {text.previewReady}</div>
              </div>
              <button type="button" onClick={() => { setFile(null); setPreview(""); setResult(null); }} className="mt-3 w-full rounded-2xl py-2 text-sm font-bold text-emerald-700 transition hover:bg-emerald-50">{text.changePhoto}</button>
            </div>
          ) : (
            <div className="grid gap-3">
              <button type="button" onClick={() => setCameraOpen(true)} className="group flex min-h-28 items-center gap-4 rounded-[1.5rem] bg-gradient-to-br from-emerald-600 to-teal-500 p-5 text-left text-white shadow-lg shadow-emerald-600/20 transition active:scale-[0.98]">
                <span className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-white/15 backdrop-blur"><CameraIcon className="h-7 w-7" /></span>
                <span><span className="block text-lg font-black">{text.takePhoto}</span><span className="mt-1 block text-xs leading-5 text-white/75">{text.cameraHint}</span></span>
              </button>

              <label htmlFor="upload-input" className="group flex min-h-28 cursor-pointer items-center gap-4 rounded-[1.5rem] border-2 border-dashed border-emerald-200 bg-emerald-50/60 p-5 text-left transition hover:border-emerald-400 hover:bg-emerald-50 active:scale-[0.98]">
                <span className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-white text-emerald-700 shadow-sm"><UploadIcon className="h-7 w-7" /></span>
                <span><span className="block text-lg font-black text-emerald-950">{text.uploadPhoto}</span><span className="mt-1 block text-xs leading-5 text-slate-500">{text.uploadHint}</span></span>
              </label>
              <input id="upload-input" type="file" accept="image/*" onChange={handleUpload} className="hidden" />
            </div>
          )}

          <button type="submit" disabled={!file || isLoading} className="mt-4 flex min-h-14 w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-5 text-[15px] font-black text-white shadow-lg shadow-slate-950/15 transition active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none">
            {isLoading ? <><span className="h-5 w-5 animate-spin rounded-full border-2 border-white/25 border-t-white" />{text.classifying}</> : <><SparkIcon />{text.classify}</>}
          </button>

          {status && <p role="status" className="mt-3 rounded-xl bg-rose-50 px-4 py-3 text-center text-sm font-semibold text-rose-700">{status}</p>}
        </form>

        <PredictionResult result={result} language={language} />
        <p className="mt-7 text-center text-[11px] leading-5 text-slate-400">Kigali E-Waste Classifier · Decision support only</p>
      </main>

      {cameraOpen && <CameraModal language={language} onClose={() => setCameraOpen(false)} onCapture={handleCameraCapture} />}
    </div>
  );
}
