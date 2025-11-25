import React from "react";

const RulesCard: React.FC = () => {
  return (
    <div className="card mt-3 p-3 shadow-sm text-start">
      <h5>Interview Rules:</h5>
      <ul>
        <li>Stay in fullscreen mode during the interview.</li>
        <li>Do not switch tabs or open other windows.</li>
        <li>Only one face should be visible in the camera.</li>
        <li>Avoid typing unless instructed.</li>
      </ul>
    </div>
  );
};

export default RulesCard;
