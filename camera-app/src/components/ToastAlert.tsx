import React from "react";

interface ToastAlertProps {
  messages: string[];
  onClear: () => void;
}

const ToastAlert: React.FC<ToastAlertProps> = ({ messages, onClear }) => {
  if (messages.length === 0) return null;

  return (
    <div className="fixed top-5 right-5 z-[9999] flex flex-col gap-3">
      {messages.map((msg, idx) => (
        <div
          key={idx}
          className="fade-in bg-red-500/90 backdrop-blur-md text-white px-6 py-4 rounded-lg shadow-lg flex items-center justify-between min-w-[300px] border border-white/20"
        >
          <span className="font-medium">{msg}</span>
          <button
            onClick={onClear}
            className="bg-transparent border-none text-white text-xl cursor-pointer pl-4 opacity-80 hover:opacity-100 transition-opacity"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  );
};

export default ToastAlert;
