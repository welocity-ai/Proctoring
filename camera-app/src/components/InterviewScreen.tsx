import React from "react";

interface InterviewScreenProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  elapsed: number;
  onStopInterview: () => void;
  onExitFullscreen: () => void;
}

const InterviewScreen: React.FC<InterviewScreenProps> = ({
  videoRef,
  elapsed,
  onStopInterview,
  onExitFullscreen,
}) => {
  // Format seconds to MM:SS
  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex flex-col items-center w-full h-full fade-in">
      {/* Header / Top Bar */}
      <div className="card w-full flex items-center justify-between mb-6 !p-4">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)] animate-pulse"></div>
          <span className="font-semibold text-lg">Live Proctoring</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-text-secondary">Time Elapsed:</span>
          <span className="font-mono text-xl font-bold text-text-primary">
            {formatTime(elapsed)}
          </span>
        </div>
        <button className="btn btn-danger py-2 px-4 text-sm" onClick={onStopInterview}>
          End Interview
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex items-center justify-center w-full flex-1 relative min-h-[60vh]">
        <div className="relative rounded-2xl overflow-hidden shadow-2xl border-2 border-white/10 bg-black w-full h-full max-w-full max-h-[80vh] aspect-video">
           <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          ></video>
        </div>
        
        {/* Overlay Controls */}
        <div className="absolute bottom-8 right-8 z-10">
             <button className="btn btn-outline-primary bg-bg-surface/50 backdrop-blur-sm border-white/20 text-white hover:bg-white/10" onClick={onExitFullscreen}>
              Exit Fullscreen
            </button>
        </div>
      </div>

      <div className="mt-6 text-text-secondary text-sm">
        <p>Monitoring active. Please stay focused on the screen.</p>
      </div>
    </div>
  );
};

export default InterviewScreen;
