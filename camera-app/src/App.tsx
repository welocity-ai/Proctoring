import React, { useRef, useState } from "react";
import CameraPreview from "./components/CameraPreview";
import InterviewScreen from "./components/InterviewScreen";
import DoneScreen from "./components/DoneScreen";
import ToastAlert from "./components/ToastAlert";
import { useProctoring } from "./hooks/useProctoring";

const BACKEND_WS = "ws://localhost:8000/ws";
const BACKEND_REPORT = "http://localhost:8000/generate_report";

type Phase = "preview" | "interview" | "done";

const App: React.FC = () => {
  const [phase, setPhase] = useState<Phase>("preview");
  const [sessionId] = useState(() => Math.random().toString(36).slice(2, 10));
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const {
    connectWS,
    startTimer,
    stopTimer,
    cleanup,
    connected,
    elapsed,
    flags,
    setFlags,
    startMonitoring,
  } = useProctoring(sessionId, BACKEND_WS);

  const enterFullscreen = async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    }
  };

  const exitFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    }
  };

  const beginInterview = async () => {
    setPhase("interview");
    connectWS();
    startTimer();
    startMonitoring();
    await enterFullscreen();

    // Attach same preview stream
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
      await videoRef.current.play();
    }
  };

  const stopInterview = async () => {
    stopTimer();
    exitFullscreen();
    cleanup();

    // Stop camera and mic immediately
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      setCameraStream(null);
      if (videoRef.current) videoRef.current.srcObject = null;
    }

    setPhase("done");

    // Generate report and download
    const res = await fetch(`${BACKEND_REPORT}/${sessionId}`, { method: "POST" });
    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${sessionId}_report.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      console.log("✅ Report downloaded successfully!");
    } else {
      console.error("Failed to generate report");
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-8 flex flex-col items-center justify-center flex-1 fade-in">
      {phase === "preview" && (
        <CameraPreview
          videoRef={videoRef}
          setCameraStream={setCameraStream}
          onStartInterview={beginInterview}
        />
      )}

      {phase === "interview" && (
        <InterviewScreen
          videoRef={videoRef}
          elapsed={elapsed}
          onStopInterview={stopInterview}
          onExitFullscreen={exitFullscreen}
        />
      )}

      {phase === "done" && <DoneScreen />}

      <ToastAlert messages={flags} onClear={() => setFlags([])} />

      <div className="mt-8 bg-bg-surface px-4 py-2 rounded-full text-sm text-text-secondary border border-white/10 inline-flex items-center gap-2">
        <span>Session: <span className="font-mono text-text-primary">{sessionId}</span></span>
        <span className="text-white/20">|</span>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${connected ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" : "bg-red-500"}`} />
          {connected ? "Connected" : "Disconnected"}
        </div>
      </div>
    </div>
  );
};

export default App;
