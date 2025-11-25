import React from "react";

const DoneScreen: React.FC = () => {
  return (
    <div className="card flex flex-col items-center gap-6 fade-in text-center w-full max-w-md">
      <div className="text-6xl animate-bounce">🎉</div>
      <h2 className="text-green-500 text-3xl font-bold">Interview Completed!</h2>
      <p className="text-text-secondary max-w-sm mx-auto">
        Thank you for completing the interview. Your session has been recorded and the report is being generated.
      </p>
      <div className="mt-4 p-4 bg-bg-surface rounded-lg border border-white/5 w-full">
        <p className="mb-0 text-sm text-text-primary">You may now close this tab.</p>
      </div>
    </div>
  );
};
export default DoneScreen;
