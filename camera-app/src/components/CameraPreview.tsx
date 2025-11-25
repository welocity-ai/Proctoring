// src/components/CameraPreview.tsx
import React, { useState } from "react";

interface CameraPreviewProps {
  onStartInterview: () => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  setCameraStream: (stream: MediaStream | null) => void;
}

const CameraPreview: React.FC<CameraPreviewProps> = ({
  onStartInterview,
  videoRef,
  setCameraStream,
}) => {
  const [previewActive, setPreviewActive] = useState(false);

  const handlePreview = async () => {
    try {
      // Ask for permissions
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setCameraStream(stream);
      setPreviewActive(true);
      console.log("🎥 Camera + 🎤 Microphone access granted");
    } catch (err) {
      console.error("❌ Failed to access camera/microphone:", err);
      alert("Please allow access to your camera and microphone to continue.");
    }
  };

  const handleStopPreview = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setCameraStream(null);
      setPreviewActive(false);
      console.log("🛑 Camera & Microphone stopped");
    }
  };

  return (
    <div className="card flex flex-col items-center gap-6 fade-in w-full max-w-2xl">
      <div className="text-center space-y-2">
        <h3 className="text-2xl">AI Proctoring — Preview</h3>
        <p className="text-text-secondary max-w-lg mx-auto">
          Click “Preview Camera” to allow and test your camera and microphone.
          Then start your interview.
        </p>
      </div>

      <div className="relative rounded-xl overflow-hidden shadow-lg border-2 border-white/10 bg-black w-[320px] h-[180px]">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
        ></video>
      </div>

      <div className="flex items-center gap-4 mt-2">
        {!previewActive ? (
          <button className="btn btn-outline-primary" onClick={handlePreview}>
            🎥 Preview Camera
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleStopPreview}>
            🛑 Stop Preview
          </button>
        )}

        <button
          className="btn btn-success"
          onClick={onStartInterview}
          disabled={!previewActive}
        >
          🚀 Start Interview
        </button>
      </div>
    </div>
  );
};

export default CameraPreview;
