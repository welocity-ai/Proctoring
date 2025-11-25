// src/hooks/useProctoring.ts
import { useEffect, useRef, useState } from "react";

interface SessionEvent {
  type: string;
  event?: string;
  b64?: string;
  ts?: number;
  duration?: number;
}

export function useProctoring(
  sessionId: string,
  backendWS: string
) {
  const wsRef = useRef<WebSocket | null>(null);
  const connectedRef = useRef(false);
  const [connected, setConnected] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [flags, setFlags] = useState<string[]>([]);
  const timerInterval = useRef<number | null>(null);

  // keep ref to monitoring cleanup so cleanup() can remove listeners
  const monitoringCleanupRef = useRef<(() => void) | null>(null);

  const nowSec = () => Math.floor(Date.now() / 1000);

  // WebSocket connect
  const connectWS = () => {
    if (wsRef.current) return;
    const ws = new WebSocket(`${backendWS}/${sessionId}`);
    ws.onopen = () => {
      connectedRef.current = true;
      setConnected(true);
      console.log("[WS] connected");
    };
    ws.onclose = () => {
      connectedRef.current = false;
      setConnected(false);
      wsRef.current = null;
      console.log("[WS] closed");
    };
    ws.onerror = (e) => console.error("[WS] error", e);
    wsRef.current = ws;
  };

  // safe send (only sends if open)
  const safeSend = (payload: SessionEvent) => {
    if (connectedRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(payload));
      } catch (e) {
        console.error("safeSend failed", e);
      }
    } else {
      // not connected — skip (we are not queueing frames now)
      console.warn("WS not open; skipping send", payload);
    }
  };

  // Flag push with cooldown
  const flagCooldown = useRef<Record<string, number>>({});
  const pushFlag = (msg: string, cooldownSec = 5) => {
    const now = Date.now();
    const last = flagCooldown.current[msg] || 0;
    if (now - last < cooldownSec * 1000) return;
    flagCooldown.current[msg] = now;
    setFlags((prev) => [...prev, msg]);

    // dispatch a UI event so InterviewScreen shows the banner (auto-dismiss handled there)
    window.dispatchEvent(new CustomEvent("showCaution", { detail: { msg } }));
  };

  // Timer
  const startTimer = () => {
    if (timerInterval.current) return;
    timerInterval.current = window.setInterval(() => setElapsed((e) => e + 1), 1000);
  };
  const stopTimer = () => {
    if (timerInterval.current) {
      clearInterval(timerInterval.current);
      timerInterval.current = null;
    }
  };

  // Monitoring: keyboard, visibility, fullscreen
  const startMonitoring = () => {
    // keyboard
    const onKeyDown = (e: KeyboardEvent) => {
      if (!e.key) return;
      const ignored = ["Shift", "Alt", "Meta", "Control", "Tab", "Escape"];
      if (ignored.includes(e.key)) return;
      pushFlag("⚠️ Keyboard activity detected", 5);
      safeSend({ type: "event", event: "keyboard", ts: nowSec() });
    };

    // Continuous checks for state-based violations (Fullscreen only)
    // Tab switch is handled via visibilitychange events to capture accurate duration
    const violationInterval = setInterval(() => {
      // Fullscreen Exit
      if (!document.fullscreenElement) {
        pushFlag("⚠️ Exited fullscreen", 5);
        safeSend({ type: "event", event: "exit_fullscreen", ts: nowSec() });
      }
    }, 1000);

    // Tab Switch Logic
    let tabSwitchStart = 0;
    const onVisibilityChange = () => {
      if (document.hidden) {
        // User switched away
        tabSwitchStart = Date.now();
        pushFlag("⚠️ Tab switching detected", 5);
        safeSend({ type: "event", event: "tab_switch", ts: nowSec() });
      } else {
        // User returned
        if (tabSwitchStart > 0) {
          const duration = Math.round((Date.now() - tabSwitchStart) / 1000);
          if (duration > 0) {
            safeSend({ type: "event", event: "tab_switch", ts: nowSec(), duration });
          }
          tabSwitchStart = 0;
        }
      }
    };

    const onFullscreenChange = () => {
      if (!document.fullscreenElement) pushFlag("⚠️ Candidate exited the fullscreen mode", 5);
    };

    window.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("visibilitychange", onVisibilityChange, true);
    document.addEventListener("fullscreenchange", onFullscreenChange, true);

    monitoringCleanupRef.current = () => {
      clearInterval(violationInterval);
      window.removeEventListener("keydown", onKeyDown, true);
      document.removeEventListener("visibilitychange", onVisibilityChange, true);
      document.removeEventListener("fullscreenchange", onFullscreenChange, true);
    };
  };

  // cleanup will stop timer, close ws, and also remove monitoring listeners
  const cleanup = () => {
    stopTimer();
    try {
      monitoringCleanupRef.current?.();
      monitoringCleanupRef.current = null;
    } catch (e) {
      console.warn("monitoring cleanup error", e);
    }
    try {
      wsRef.current?.close();
      wsRef.current = null;
      connectedRef.current = false;
      setConnected(false);
    } catch (e) {
      // ignore if already closed
    }
  };

  useEffect(() => {
    // on unmount ensure cleanup
    return () => cleanup();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    connectWS,
    startTimer,
    stopTimer,
    cleanup,
    connected,
    elapsed,
    flags,
    setFlags,
    startMonitoring,
  };
}
