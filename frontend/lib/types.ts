export type JointName =
  | "base"
  | "shoulder"
  | "elbow"
  | "wrist_vertical"
  | "wrist_rotation"
  | "gripper";

export interface ObservationItem {
  id: string;
  name: string;
  bbox: number[];
  confidence: number;
  uncertain: boolean;
  zone: string | null;
  held: boolean;
  source: string;
  localization_note: string;
  color: string;
}

export interface PlanStep {
  tool: string;
  label: string;
  detail: string;
}

export interface ArmState {
  robot: {
    execution_mode: "dry_run" | "armed";
    serial_connected: boolean;
    serial_mode: string;
    port: string;
    synced: boolean;
    joints_known: boolean;
    joint_source: string;
    moving: boolean;
    activity: string;
    angles: Record<JointName, number | null>;
    needs_status_sync: boolean;
    unexpected_reset: boolean;
    last_error: string;
    pose_errors: string[];
    commands_sent: number;
    uart: string[];
  };
  observation: {
    mode: string;
    camera_index: number;
    camera_label: string;
    camera_status: string;
    stale: boolean;
    stale_reason: string;
    video_stale: boolean;
    video_error: string;
    selected_id: string | null;
    seen?: string;
    observation_ts?: number;
    items: ObservationItem[];
    frame_ts: number;
  };
  enose: { mode: string; connected: boolean; detail: string; base_url: string };
  samples?: {
    ready: boolean;
    error: string;
    items: SmellRecord[];
  };
  mission: {
    status: string;
    phase: string;
    plan: null | {
      id: string;
      summary: string;
      final_state: string;
      planner: string;
      planner_note: string;
      enose_note: string;
      steps: PlanStep[];
    };
    clarification: null | {
      question: string;
      options: Array<{ id: string; label: string }>;
      expects_text?: boolean;
    };
    timeline: Array<{ id: number; ts: number; phase: string; level: string; message: string }>;
    record: { request: string; selected: { id: string; name: string } | null; completed: string[] };
    last_stop: null | {
      software_stop: boolean;
      uart_sent: boolean;
      acknowledged: boolean;
      detail: string;
    };
    error: string;
    outcome: string;
    running: boolean;
  };
  settings: {
    execution_mode: string;
    serial_mode: string;
    serial_port: string;
    camera_mode: string;
    camera_index: number;
    llm_enabled: boolean;
    llm_base_url: string;
    llm_model: string;
    vlm_enabled: boolean;
    vlm_base_url: string;
    vlm_model: string;
    enose_mode: string;
    enose_base_url: string;
    baud: number;
    aim_offset_x?: number;
    aim_offset_y?: number;
    aim_offset_z?: number;
  };
}

export interface SmellRecord {
  id: number;
  created_at: string;
  object_name: string;
  object_id: string;
  is_container: boolean;
  contents: string;
  channels: number[];
  source: string;
}

export interface DeviceList {
  cameras: Array<{ id: string; label: string }>;
  ports: Array<{ device: string; description: string }>;
  scanning?: boolean;
  camera_error?: string;
}

export const JOINTS: Array<{ key: JointName; label: string }> = [
  { key: "base", label: "Base" },
  { key: "shoulder", label: "Shoulder" },
  { key: "elbow", label: "Elbow" },
  { key: "wrist_vertical", label: "Wrist vertical" },
  { key: "wrist_rotation", label: "Wrist rotation" },
  { key: "gripper", label: "Gripper" },
];

export const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";
