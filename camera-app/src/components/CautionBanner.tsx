import React from "react";

interface CautionBannerProps {
  message: string;
  color?: string;
}

const CautionBanner: React.FC<CautionBannerProps> = ({ message, color = "#ffc107" }) => (
  <div
    style={{
      position: "fixed",
      bottom: 0,
      left: 0,
      width: "100%",
      backgroundColor: color,
      color: "#000",
      textAlign: "center",
      padding: "8px 12px",
      fontWeight: 500,
      fontSize: "15px",
      zIndex: 10000,
      boxShadow: "0 -2px 6px rgba(0,0,0,0.1)",
    }}
  >
    {message}
  </div>
);

export default CautionBanner;
