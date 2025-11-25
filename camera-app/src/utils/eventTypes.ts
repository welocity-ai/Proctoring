export interface SessionEvent {
    ts: number;
    type: string;
    details?: Record<string, any>;
    screenshot?: string;
  }
  
  export interface Report {
    session_id: string;
    start_time: number;
    end_time: number;
    total_flags: number;
    events: SessionEvent[];
  }
  