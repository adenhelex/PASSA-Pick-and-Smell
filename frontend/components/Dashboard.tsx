"use client";

import { useEffect, useRef, useState } from "react";
import ArmView from "./ArmView";
import { ArmState, BACKEND, DeviceList, JOINTS, JointName, SmellRecord } from "../lib/types";
import { VoiceLang, hasThai, listen, speak, speechSupported, voiceLooksWrong } from "../lib/voice";

const PHASES = [
  ["scanning", "Scanning"],
  ["target_found", "Target found"],
  ["approaching", "Approaching"],
  ["grasping", "Grasping"],
  ["sampling", "Sampling"],
  ["placing", "Placing"],
  ["complete", "Complete"],
];

function hostFromBase(url: string) {
  try {
    const parsed = new URL(url);
    if (parsed.port && parsed.port !== "11434") return `${parsed.hostname}:${parsed.port}`;
    return parsed.hostname;
  } catch {
    return url.replace(/^https?:\/\//, "");
  }
}

function ollamaBase(value: string) {
  const text = value.trim().replace(/\/$/, "");
  if (!text) return "http://127.0.0.1:11434";
  if (text.startsWith("http://") || text.startsWith("https://")) {
    try {
      const parsed = new URL(text);
      return `${parsed.protocol}//${parsed.hostname}:${parsed.port || "11434"}`;
    } catch {
      return text;
    }
  }
  return text.includes(":") ? `http://${text}` : `http://${text}:11434`;
}

function csvCell(value: string | number | boolean) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function exportSmells(items: SmellRecord[]) {
  const header = [
    "id",
    "created_at",
    "object_name",
    "object_id",
    "is_container",
    "contents",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "source",
  ];
  const lines = items.map((item) => {
    const channels = Array.from({ length: 8 }, (_, index) => item.channels[index] ?? 0);
    return [
      item.id,
      item.created_at,
      item.object_name,
      item.object_id,
      item.is_container ? "yes" : "no",
      item.contents,
      ...channels,
      item.source,
    ]
      .map(csvCell)
      .join(",");
  });
  const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "passa-smells.csv";
  link.click();
  URL.revokeObjectURL(url);
}

export default function Dashboard() {
  const [state, setState] = useState<ArmState | null>(null);
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [devices, setDevices] = useState<DeviceList>({ cameras: [], ports: [] });
  const [cameraScan, setCameraScan] = useState(0);
  const [draft, setDraft] = useState<ArmState["settings"] | null>(null);
  const [llmHost, setLlmHost] = useState("");
  const [llmHostReady, setLlmHostReady] = useState(false);
  const [llmModels, setLlmModels] = useState<string[]>([]);
  const [ollamaHosts, setOllamaHosts] = useState<Array<{ ip: string; base_url: string; models: string[] }>>([]);
  const [llmScanning, setLlmScanning] = useState(false);
  const [llmScan, setLlmScan] = useState(0);
  const [llmNote, setLlmNote] = useState("");
  const [notice, setNotice] = useState("");
  const [commandText, setCommandText] = useState("");
  const [inside, setInside] = useState("");
  const [voiceLang, setVoiceLang] = useState<VoiceLang>("th-TH");
  const [voiceLoop, setVoiceLoop] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceNote, setVoiceNote] = useState("");
  const uartRef = useRef<HTMLDivElement>(null);
  const spokenQuestion = useRef("");
  const voiceLoopRef = useRef(false);
  const voiceLangRef = useRef<VoiceLang>("th-TH");
  const listeningStop = useRef<(() => void) | null>(null);
  const askCycle = useRef(0);
  const wasRunning = useRef(false);
  const askForVoiceCommandRef = useRef<() => Promise<void>>(async () => undefined);

  useEffect(() => {
    let source: EventSource | null = null;
    const apply = (next: ArmState) => {
      setState(next);
      setOffline(false);
      setDraft((current) => current ?? next.settings);
    };
    const load = () =>
      fetch(`${BACKEND}/api/state`)
        .then((response) => {
          if (!response.ok) throw new Error("backend");
          return response.json();
        })
        .then(apply)
        .catch(() => setOffline(true));
    load();
    source = new EventSource(`${BACKEND}/api/events`);
    source.onmessage = (event) => apply(JSON.parse(event.data));
    source.onerror = () => setOffline(true);
    const poll = window.setInterval(load, 4000);
    return () => {
      source?.close();
      window.clearInterval(poll);
    };
  }, []);

  useEffect(() => {
    voiceLoopRef.current = voiceLoop;
  }, [voiceLoop]);

  useEffect(() => {
    voiceLangRef.current = voiceLang;
  }, [voiceLang]);

  useEffect(() => {
    const question = state?.mission.clarification?.question?.trim() || "";
    if (!question) {
      spokenQuestion.current = "";
      return;
    }
    if (spokenQuestion.current === question) return;
    spokenQuestion.current = question;
    const line = /can't detect|cannot see|did not (answer|choose)|ตรวจไม่พบ|ไม่เห็น/i.test(question)
      ? question
      : `I can't detect that object. ${question}`;
    void speak(line, hasThai(line) ? "th-TH" : "en-US");
  }, [state?.mission.clarification?.question]);

  useEffect(() => {
    let cancel = false;
    let timer = 0;
    const pull = (refresh: boolean) => {
      fetch(`${BACKEND}/api/devices${refresh ? "?refresh=1" : ""}`)
        .then((response) => response.json())
        .then((data: DeviceList) => {
          if (cancel) return;
          setDevices(data);
          if (data.scanning) timer = window.setTimeout(() => pull(false), 1200);
        })
        .catch(() => undefined);
    };
    pull(cameraScan > 0);
    return () => {
      cancel = true;
      window.clearTimeout(timer);
    };
  }, [cameraScan]);

  useEffect(() => {
    if (!draft || llmHostReady) return;
    setLlmHost(hostFromBase(draft.llm_base_url));
    setLlmHostReady(true);
  }, [draft, llmHostReady]);

  useEffect(() => {
    const match = ollamaHosts.find((host) => host.ip === llmHost);
    if (match) setLlmModels(match.models);
  }, [ollamaHosts, llmHost]);

  useEffect(() => {
    let cancel = false;
    let timer = 0;
    const pull = (refresh: boolean) => {
      fetch(`${BACKEND}/api/llm/scan${refresh ? "?refresh=1" : ""}`)
        .then((response) => response.json())
        .then((data: { scanning?: boolean; hosts?: Array<{ ip: string; base_url: string; models: string[] }>; error?: string }) => {
          if (cancel) return;
          const hosts = data.hosts ?? [];
          setOllamaHosts(hosts);
          setLlmScanning(Boolean(data.scanning));
          if (data.scanning) {
            setLlmNote("Scanning this network for Ollama on port 11434.");
            timer = window.setTimeout(() => pull(false), 1000);
          } else {
            setLlmNote(hosts.length ? "" : data.error || "No Ollama server answered on this network.");
          }
        })
        .catch(() => {
          if (!cancel) setLlmNote("Could not scan for Ollama.");
        });
    };
    pull(true);
    return () => {
      cancel = true;
      window.clearTimeout(timer);
    };
  }, [llmScan]);

  useEffect(() => {
    const node = uartRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [state?.robot.uart]);

  async function post(path: string, body?: unknown) {
    const response = await fetch(`${BACKEND}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setNotice(payload.detail || "The backend refused that action.");
      return null;
    }
    if (payload.robot && payload.mission) setState(payload);
    if (payload.state) setState(payload.state);
    return payload;
  }

  async function call(path: string, body?: unknown) {
    setBusy(true);
    setNotice("");
    try {
      return await post(path, body);
    } catch {
      setOffline(true);
      setNotice("The Python backend is not reachable.");
    } finally {
      setBusy(false);
    }
  }

  async function startMission(path: string, body?: unknown) {
    setBusy(true);
    setNotice("");
    try {
      const payload = await post(path, body);
      const missionState = payload?.state?.mission ?? payload?.mission;
      if (missionState?.plan?.intent === "noop" && missionState.plan.summary) {
        const line = String(missionState.plan.summary);
        setNotice(line);
        void speak(line, hasThai(line) ? "th-TH" : voiceLangRef.current);
      }
      if (missionState?.status === "ready" && missionState.plan && !missionState.running) {
        await post("/api/run");
      }
      return payload;
    } catch {
      setOffline(true);
      setNotice("The Python backend is not reachable.");
      return null;
    } finally {
      setBusy(false);
    }
  }

  function stopListening() {
    listeningStop.current?.();
    listeningStop.current = null;
    setListening(false);
  }

  async function askForVoiceCommand() {
    if (!voiceLoopRef.current) return;
    const cycle = ++askCycle.current;
    const lang = voiceLangRef.current;
    stopListening();
    setVoiceNote(lang === "th-TH" ? "กำลังขอคำสั่ง..." : "Asking for a command...");
    // Speak only one language before listening. Mixing English into the Thai
    // prompt makes Chrome's Thai speech recognition less reliable.
    const prompt = lang === "th-TH" ? "กรุณาออกคำสั่งให้หุ่นยนต์ได้เลย" : "Please command the robot.";
    await speak(prompt, lang);
    if (!voiceLoopRef.current || cycle !== askCycle.current) return;
    if (!speechSupported()) {
      setVoiceNote("This browser cannot listen. Use Chrome, or type the command.");
      return;
    }
    setListening(true);
    setVoiceNote(lang === "th-TH" ? "กำลังฟังภาษาไทย... พูดคำสั่งได้เลย" : "Listening... say the command.");
    const session = listen(lang, 10000, lang === "th-TH" ? 1000 : 800);
    listeningStop.current = session.stop;
    try {
      const heard = (await session.promise).trim();
      if (!voiceLoopRef.current || cycle !== askCycle.current) return;
      if (!heard || voiceLooksWrong(lang, heard)) {
        setVoiceNote(
          lang === "th-TH"
            ? `ไม่ได้ยินภาษาไทยชัดเจน${heard ? ` (ได้ยินว่า: ${heard})` : ""}. พูดอีกครั้ง`
            : "No clear command heard. Trying again.",
        );
        setListening(false);
        await speak(lang === "th-TH" ? "ไม่ได้ยินภาษาไทยชัดเจน กรุณาพูดอีกครั้ง" : "I did not hear a clear command. Please say it again.", lang);
        window.setTimeout(() => {
          if (voiceLoopRef.current) void askForVoiceCommandRef.current();
        }, 500);
        return;
      }
      setCommandText(heard);
      setVoiceNote(heard);
      setListening(false);
      await speak(lang === "th-TH" ? `ได้ยินว่า ${heard}` : `I heard ${heard}`, lang);
      if (!voiceLoopRef.current || cycle !== askCycle.current) return;
      await startMission("/api/command", { text: heard });
    } catch (error) {
      if (!voiceLoopRef.current || cycle !== askCycle.current) return;
      setListening(false);
      setVoiceNote(error instanceof Error ? error.message : "Microphone failed.");
      window.setTimeout(() => {
        if (voiceLoopRef.current) void askForVoiceCommandRef.current();
      }, 1200);
    }
  }

  askForVoiceCommandRef.current = askForVoiceCommand;

  function startVoiceLoop() {
    if (!speechSupported()) {
      setVoiceNote("This browser cannot listen. Use Chrome for Thai and English voice commands.");
      return;
    }
    setVoiceLoop(true);
    voiceLoopRef.current = true;
    void askForVoiceCommandRef.current();
  }

  function endVoiceLoop() {
    askCycle.current += 1;
    setVoiceLoop(false);
    voiceLoopRef.current = false;
    stopListening();
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    setVoiceNote("");
  }

  function changeVoiceLang(next: VoiceLang) {
    if (next === voiceLang) return;
    // Stop any active listen/speak cycle before switching language.
    endVoiceLoop();
    setVoiceLang(next);
    voiceLangRef.current = next;
    setVoiceNote(next === "th-TH" ? "เปลี่ยนเป็นภาษาไทยแล้ว กด Run loop เมื่อพร้อม" : "Switched to English. Press Run loop when ready.");
  }

  const mission = state?.mission;
  const robot = state?.robot;
  const active = Boolean(mission?.running || mission?.status === "awaiting_clarification");
  const armed = robot?.execution_mode === "armed";

  useEffect(() => {
    const running = Boolean(mission?.running);
    if (running) {
      wasRunning.current = true;
      return;
    }
    if (!voiceLoopRef.current || !wasRunning.current) return;
    wasRunning.current = false;
    if (mission?.status === "awaiting_clarification") return;
    const timer = window.setTimeout(() => {
      if (voiceLoopRef.current) void askForVoiceCommandRef.current();
    }, 900);
    return () => window.clearTimeout(timer);
  }, [mission?.running, mission?.status]);

  function jog(joint: JointName, delta: number) {
    call("/api/jog", { joint, delta });
  }

  async function chooseCamera(value: string, label = "") {
    if (!draft) return;
    const previous = { camera_mode: draft.camera_mode, camera_index: draft.camera_index };
    const camera_mode = value === "mock" ? "mock" : "hardware";
    const camera_index = value === "mock" ? draft.camera_index : Number(value);
    setDraft({ ...draft, camera_mode, camera_index });
    const payload = await call("/api/settings", {
      camera_mode,
      camera_index,
      camera_label: label || undefined,
    });
    if (!payload?.settings) {
      setDraft((current) => (current ? { ...current, ...previous } : current));
    }
  }

  async function usePlanner(enabled: boolean, model = draft?.llm_model ?? "", host = llmHost) {
    if (!draft) return;
    const previous = {
      llm_base_url: draft.llm_base_url,
      llm_model: draft.llm_model,
      llm_enabled: draft.llm_enabled,
    };
    const llm_base_url = ollamaBase(host);
    setDraft({ ...draft, llm_base_url, llm_model: model, llm_enabled: enabled });
    const payload = await call("/api/settings", { llm_base_url, llm_model: model, llm_enabled: enabled });
    if (!payload?.settings) {
      setDraft((current) => (current ? { ...current, ...previous } : current));
    }
  }

  const serverChoices = ollamaHosts.some((host) => host.ip === llmHost) || !llmHost
    ? ollamaHosts
    : [{ ip: llmHost, base_url: ollamaBase(llmHost), models: llmModels }, ...ollamaHosts];

  function chooseServer(ip: string) {
    const chosen = serverChoices.find((host) => host.ip === ip);
    if (!chosen || !draft) return;
    setLlmHost(chosen.ip);
    setLlmModels(chosen.models);
    const preferred = chosen.models.find((name) => name.startsWith("llama3.2"))
      || chosen.models.find((name) => !name.startsWith("llava"))
      || chosen.models[0]
      || draft.llm_model;
    const model = chosen.models.includes(draft.llm_model) ? draft.llm_model : preferred;
    if (draft.llm_enabled) usePlanner(true, model, chosen.ip);
    else setDraft({ ...draft, llm_base_url: chosen.base_url, llm_model: model });
  }

  const modelChoices = draft
    ? llmModels.includes(draft.llm_model)
      ? llmModels
      : [draft.llm_model, ...llmModels].filter(Boolean)
    : llmModels;
  const cameras = devices.cameras.length ? devices.cameras : [{ id: "mock", label: "Mock tabletop camera" }];
  const selectedCamera = draft ? (draft.camera_mode === "hardware" ? String(draft.camera_index) : "mock") : "mock";
  const cameraOptions = cameras.some((camera) => camera.id === selectedCamera)
    ? cameras
    : [...cameras, { id: selectedCamera, label: state?.observation.camera_label || `Camera ${selectedCamera}` }];

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <strong>PASSA</strong>
          <span>Pick and smell</span>
        </div>
        <span className={armed ? "mode-pill armed" : "mode-pill dry"}>{armed ? "Armed" : "Dry-run"}</span>
        <span className="link-pill">
          {robot?.serial_connected ? "Serial connected" : "Serial offline"} · {robot?.activity === "moving" ? "Moving" : "Idle"}
        </span>
        <div className="top-spacer" />
        <button
          className={mission?.running || voiceLoop ? "stop live" : "stop"}
          onClick={() => {
            endVoiceLoop();
            call("/api/stop");
          }}
          disabled={!state}
        >
          Stop
        </button>
      </header>
      {mission?.last_stop && <p className="stop-banner">{mission.last_stop.detail}</p>}
      {offline && (
        <p className="offline">Backend unreachable at {BACKEND}. Start the Python server, then refresh.</p>
      )}
      <main className="stage">
        <section className="panel arm-panel">
          <header>
            Arm
            <span>
              {robot?.joints_known
                ? robot.activity === "moving"
                  ? "Moving"
                  : "Commanded pose"
                : "Joints unknown"}
            </span>
          </header>
          <ArmView
            angles={robot?.angles ?? { base: null, shoulder: null, elbow: null, wrist_vertical: null, wrist_rotation: null, gripper: null }}
            jointsKnown={Boolean(robot?.joints_known)}
            items={state?.observation.items ?? []}
            selectedId={state?.observation.selected_id ?? null}
            phase={mission?.phase ?? ""}
            showDefaultScene={(state?.observation.mode ?? "mock") !== "hardware"}
          />
        </section>
        <section className="panel records">
          <header>
            Saved smells
            <span className="record-actions">
              <button
                type="button"
                className="tiny"
                disabled={!state?.samples?.items.length}
                onClick={() => exportSmells(state?.samples?.items ?? [])}
              >
                Export CSV
              </button>
              <span>{state?.samples?.ready ? `${state.samples.items.length} saved` : "Database offline"}</span>
            </span>
          </header>
          <div className="body">
            {!state?.samples && <p className="note">Waiting for the database.</p>}
            {state?.samples && !state.samples.ready && (
              <p className="note">{state.samples.error || "PostgreSQL is not connected."}</p>
            )}
            {state?.samples?.ready && state.samples.items.length === 0 && <p className="note">Nothing saved yet.</p>}
            <ul>
              {state?.samples?.items.map((item) => (
                <li key={item.id}>
                  <div>
                    <b>{item.object_name}</b>
                    <span>{item.is_container ? item.contents || "Contents unknown" : "Not a container"}</span>
                    <time>{new Date(item.created_at).toLocaleString([], { hour12: false })}</time>
                  </div>
                  <p>
                    {item.channels.map((value, index) => (
                      <em key={index}>
                        R{index} <strong>{value.toFixed(2)}</strong>
                      </em>
                    ))}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        </section>
        <section className="panel">
          <header>
            Camera
            <span>{cameraOptions.find((camera) => camera.id === selectedCamera)?.label || "Mock scene"}</span>
          </header>
          <div className="camera-bar">
            <select
              aria-label="Camera"
              value={selectedCamera}
              disabled={busy || active || !draft}
              onChange={(event) => chooseCamera(event.target.value, event.target.selectedOptions[0]?.label ?? "")}
            >
              {cameraOptions.map((camera) => (
                <option key={camera.id} value={camera.id}>{camera.label}</option>
              ))}
            </select>
            <button type="button" disabled={busy || active || devices.scanning} onClick={() => setCameraScan((value) => value + 1)}>
              {devices.scanning ? "Looking…" : "Refresh"}
            </button>
          </div>
          {(state?.observation.camera_status || state?.observation.video_error || devices.camera_error) && (
            <p className={state?.observation.video_error || devices.camera_error ? "camera-error" : "camera-status"}>
              {state?.observation.video_error || state?.observation.camera_status || devices.camera_error}
            </p>
          )}
          <div className="camera-wrap">
            <div
              className="frame"
              onClick={(event) => {
                if (busy || mission?.running) return;
                const rect = event.currentTarget.getBoundingClientRect();
                if (rect.width < 1 || rect.height < 1) return;
                const u = ((event.clientX - rect.left) / rect.width) * 640;
                const v = ((event.clientY - rect.top) / rect.height) * 480;
                startMission("/api/select", { u, v });
              }}
            >
              <img
                key={`${state?.observation.mode}-${state?.observation.camera_index}`}
                src={`${BACKEND}/api/camera/stream?camera=${state?.observation.mode ?? "mock"}-${state?.observation.camera_index ?? 0}`}
                alt="Wrist camera preview"
              />
              {state?.observation.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={item.id === state.observation.selected_id ? "box selected" : "box"}
                  disabled={busy || Boolean(mission?.running)}
                  onClick={(event) => {
                    event.stopPropagation();
                    startMission("/api/select", { object_id: item.id });
                  }}
                  style={{
                    left: `${(item.bbox[0] / 640) * 100}%`,
                    top: `${(item.bbox[1] / 480) * 100}%`,
                    width: `${(item.bbox[2] / 640) * 100}%`,
                    height: `${(item.bbox[3] / 480) * 100}%`,
                  }}
                >
                  <span>
                    {item.name}
                    {item.uncertain ? " · uncertain" : ""}
                    {item.id === state.observation.selected_id ? " · target" : ""}
                  </span>
                </button>
              ))}
              {state?.observation.stale && (
                <div className="stale">Latest observation is stale. {state.observation.stale_reason}</div>
              )}
              <div className="saw-caption">{state?.observation.seen || "Watching for objects."}</div>
            </div>
          </div>
        </section>

        <section className="panel command">
          <header>
            Command
            <span>{voiceLoop ? (listening ? "Listening" : "Voice loop") : "Thai / English"}</span>
          </header>
          <div className="body">
            <div className="llm-pick">
              <label>
                Ollama server
                <select
                  aria-label="Ollama server"
                  value={llmHost}
                  disabled={busy || active || !serverChoices.length}
                  onChange={(event) => chooseServer(event.target.value)}
                >
                  {!serverChoices.length && <option value="">Scanning…</option>}
                  {serverChoices.map((host) => (
                    <option key={host.ip} value={host.ip}>
                      {host.ip} · {host.models.length ? host.models.slice(0, 2).join(", ") : "no models"}
                      {host.models.length > 2 ? ` +${host.models.length - 2}` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Ollama model
                <select
                  aria-label="Ollama model"
                  value={draft?.llm_model ?? ""}
                  disabled={busy || active || !draft}
                  onChange={(event) => {
                    const model = event.target.value;
                    if (draft?.llm_enabled) usePlanner(true, model);
                    else if (draft) setDraft({ ...draft, llm_model: model });
                  }}
                >
                  {modelChoices.map((model) => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </label>
              <button type="button" disabled={busy || active || llmScanning} onClick={() => setLlmScan((value) => value + 1)}>
                {llmScanning ? "Scanning…" : "Scan"}
              </button>
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={Boolean(draft?.llm_enabled)}
                disabled={busy || active || !draft}
                onChange={(event) => usePlanner(event.target.checked)}
              />
              Use this Ollama for planning
            </label>
            {llmNote && <p className="warn">{llmNote}</p>}
            <div className="voice-bar">
              <label>
                Listen in
                <select
                  aria-label="Voice language"
                  value={voiceLang}
                  disabled={busy}
                  onChange={(event) => changeVoiceLang(event.target.value as VoiceLang)}
                >
                  <option value="th-TH">ไทย Thai</option>
                  <option value="en-US">English</option>
                </select>
              </label>
              <button
                type="button"
                className={listening ? "listening" : ""}
                disabled={busy || Boolean(mission?.running) || listening}
                onClick={() => {
                  if (!speechSupported()) {
                    setVoiceNote("This browser cannot listen. Use Chrome.");
                    return;
                  }
                  setListening(true);
                  setVoiceNote(voiceLang === "th-TH" ? "กำลังฟังภาษาไทย..." : "Listening...");
                  const session = listen(voiceLang, 10000, voiceLang === "th-TH" ? 1000 : 800);
                  listeningStop.current = session.stop;
                  session.promise
                    .then((heard) => {
                      const text = heard.trim();
                      if (voiceLooksWrong(voiceLang, text)) {
                        setVoiceNote(
                          voiceLang === "th-TH"
                            ? `ไม่ได้ยินภาษาไทยชัดเจน${text ? ` (ได้ยินว่า: ${text})` : ""}. กด Mic พูดอีกครั้ง`
                            : "Nothing clear heard.",
                        );
                        return;
                      }
                      setCommandText(text);
                      setVoiceNote(text);
                      void speak(voiceLang === "th-TH" ? `ได้ยินว่า ${text}` : `I heard ${text}`, voiceLang);
                    })
                    .catch((error) => setVoiceNote(error instanceof Error ? error.message : "Microphone failed."))
                    .finally(() => {
                      listeningStop.current = null;
                      setListening(false);
                    });
                }}
              >
                {listening ? "Listening…" : "Mic"}
              </button>
              {!voiceLoop ? (
                <button type="button" className="primary" disabled={busy || active || listening} onClick={startVoiceLoop}>
                  Run loop
                </button>
              ) : (
                <button type="button" className="stop-loop" disabled={busy} onClick={endVoiceLoop}>
                  End loop
                </button>
              )}
            </div>
            {voiceNote && <p className="voice-note">{voiceNote}</p>}
            <form
              onSubmit={(event) => {
                event.preventDefault();
                const text = commandText.trim();
                if (!text) return;
                startMission("/api/command", { text });
              }}
            >
              <textarea
                aria-label="Command"
                value={commandText}
                placeholder="Thai or English. Example: หยิบแก้วใกล้ขวด / pick the glass nearest the bottle"
                disabled={busy || Boolean(mission?.running)}
                onChange={(event) => setCommandText(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
              />
            </form>
            <div className="row">
              <button
                className="primary"
                disabled={busy || Boolean(mission?.running) || !commandText.trim()}
                onClick={() => startMission("/api/command", { text: commandText.trim() })}
              >
                Send
              </button>
              <span className={armed ? "run-mode armed" : "run-mode"}>{armed ? "Armed" : "Dry-run"}</span>
              <button
                disabled={busy || active}
                onClick={() => {
                  const next = armed ? "dry_run" : "armed";
                  if (next === "armed" && !window.confirm("Armed mode can send UART motion commands. Arm the controller?")) {
                    return;
                  }
                  call("/api/execution-mode", { mode: next });
                }}
              >
                {armed ? "Switch to dry-run" : "Arm"}
              </button>
            </div>
            <div className="saw-box">
              <strong>Camera saw</strong>
              <span>{state?.observation.seen || "Watching for objects."}</span>
            </div>
            <div className="choices">
              {(state?.observation.items ?? []).filter((item) => !item.held).map((item) => (
                <button
                  key={item.id}
                  className={item.id === state?.observation.selected_id ? "choice selected" : "choice"}
                  disabled={busy || Boolean(mission?.running)}
                  onClick={() => startMission("/api/select", { object_id: item.id })}
                >
                  {item.name}
                  {item.uncertain ? " · uncertain" : ""}
                </button>
              ))}
            </div>
            {notice && <p className="note">{notice}</p>}
            {mission?.error && <p className="note">{mission.error}</p>}
            {mission?.clarification && (
              <div className="question">
                <strong>{mission.clarification.question}</strong>
                {mission.clarification.expects_text ? (
                  <form
                    className="row"
                    onSubmit={(event) => {
                      event.preventDefault();
                      const text = inside.trim();
                      if (!text) return;
                      setInside("");
                      void startMission("/api/clarify", { text });
                    }}
                  >
                    <input
                      value={inside}
                      onChange={(event) => setInside(event.target.value)}
                      placeholder="What is inside?"
                    />
                    <button className="primary" type="submit">
                      Save answer
                    </button>
                  </form>
                ) : (
                  <div className="choices">
                    {mission.clarification.options.map((option) => (
                      <button key={option.id} className="choice" onClick={() => startMission("/api/clarify", { object_id: option.id })}>
                        {option.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {mission?.plan && (
              <>
                <p className="summary">{mission.plan.summary}</p>
                {mission.plan.enose_note && <p className="note">{mission.plan.enose_note}</p>}
                {mission.plan.planner_note && <p className="note">{mission.plan.planner_note}</p>}
                <div className="final">
                  <strong>Planned final state. </strong>
                  {mission.plan.final_state}
                </div>
                <ol className="steps">
                  {mission.plan.steps.map((step, index) => (
                    <li key={`${step.tool}-${index}`}>
                      <span>{index + 1}</span>
                      <div>
                        <b>{step.label}</b>
                        <p>{step.detail}</p>
                      </div>
                    </li>
                  ))}
                </ol>
              </>
            )}
          </div>
        </section>

        <section className="panel">
          <header>Mission</header>
          <div className="body">
            <div className="rail">
              {PHASES.map(([id, label]) => (
                <span key={id} className={mission?.phase === id ? "on" : ""}>
                  {label}
                </span>
              ))}
              {(mission?.phase === "stopped" || mission?.phase === "blocked") && (
                <span className="on">{mission.phase}</span>
              )}
            </div>
            <p className="record">
              {mission?.record.request ? `You asked: ${mission.record.request}` : "No request yet."}
              {mission?.record.selected ? ` · Selected ${mission.record.selected.name}` : ""}
              {mission?.record.completed.length ? ` · Done ${mission.record.completed.join(", ")}` : ""}
            </p>
            <div className="timeline">
              {(mission?.timeline ?? []).slice().reverse().map((item) => (
                <div key={item.id} className={`event ${item.level}`}>
                  <time>{new Date(item.ts * 1000).toLocaleTimeString([], { hour12: false })}</time>
                  <span>{item.message}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <div className="side">
          <section className="panel">
            <header>
              Robot
              <span>{robot?.joint_source === "unknown" ? "Unknown" : robot?.joint_source}</span>
            </header>
            <div className="body">
              <p className="meta">
                {robot?.execution_mode === "armed" ? "Armed" : "Dry-run"} ·{" "}
                {robot?.serial_connected ? "serial connected" : "serial not connected"} ·{" "}
                {robot?.activity === "moving" ? "moving" : "idle"}
                {robot?.synced ? " · status synced" : ""}
              </p>
              <p className="section-title" style={{ paddingLeft: 0 }}>
                Commanded joint angles
              </p>
              <div className="joints">
                {JOINTS.map((joint) => {
                  const angle = robot?.angles[joint.key];
                  return (
                    <div key={joint.key} className="joint">
                      <em>{joint.label}</em>
                      <strong className={angle == null ? "unknown" : ""}>
                        {angle == null ? "—" : `${Math.round(angle)}°`}
                      </strong>
                    </div>
                  );
                })}
              </div>
              {!robot?.joints_known && (
                <p className="warn">Unknown until the backend synchronizes status, or you reset the dry-run pose.</p>
              )}
              {robot?.unexpected_reset && <p className="warn">{robot.last_error}</p>}
              {robot?.needs_status_sync && robot.last_error && <p className="warn">{robot.last_error}</p>}
              <p className="section-title" style={{ paddingLeft: 0 }}>UART</p>
              <p className="meta">
                {armed
                  ? "Lines written to the Uno, one command at a time."
                  : "Lines the controller would write. Dry-run does not open the port."}
              </p>
              <div className="uart" ref={uartRef}>
                {(robot?.uart ?? []).length === 0 && <p className="meta">No UART commands yet.</p>}
                {(robot?.uart ?? []).map((line, index) => (
                  <p key={`${index}-${line}`}>&gt; {line}</p>
                ))}
              </div>
              <div className="row">
                <button disabled={busy || !robot?.serial_connected || robot.moving} onClick={() => call("/api/serial/status")}>
                  Sync status
                </button>
                <button disabled={busy || active || armed} onClick={() => call("/api/simulation/park")}>
                  Use simulated park
                </button>
              </div>
            </div>
          </section>

          <section className="panel">
            <header>Manual</header>
            <div className="body">
              {active && <p className="warn">Manual jog is disabled during an active mission.</p>}
              <div className="row">
                <button disabled={busy || active} onClick={() => call("/api/home")}>Home</button>
                <button disabled={busy || active} onClick={() => call("/api/gripper", { action: "open" })}>Gripper open</button>
                <button disabled={busy || active} onClick={() => call("/api/gripper", { action: "close" })}>Gripper close</button>
              </div>
              {JOINTS.map((joint) => (
                <div key={joint.key} className="jog">
                  <span>{joint.label}</span>
                  <span>
                    <button className="tiny" disabled={busy || active} onClick={() => jog(joint.key, -5)}>−</button>
                    <button className="tiny" disabled={busy || active} onClick={() => jog(joint.key, 5)}>+</button>
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <header>
              Devices
              <span className={`badge ${state?.enose.mode ?? "mock"}`}>E-nose {state?.enose.mode}</span>
            </header>
            <div className="body">
              <p className="meta">{state?.enose.detail}</p>
              {draft && (
                <form
                  className="form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    call("/api/settings", draft);
                  }}
                >
                  <label>
                    Camera
                    <select aria-label="Camera in devices" value={selectedCamera} disabled={busy || active} onChange={(event) => chooseCamera(event.target.value, event.target.selectedOptions[0]?.label ?? "")}>
                      {cameraOptions.map((camera) => (
                        <option key={camera.id} value={camera.id}>{camera.label}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Serial
                    <select value={draft.serial_mode} onChange={(event) => setDraft({ ...draft, serial_mode: event.target.value })}>
                      <option value="mock">Mock Arduino</option>
                      <option value="hardware">Arduino Uno</option>
                    </select>
                  </label>
                  <label>
                    Serial port
                    <select value={draft.serial_port} onChange={(event) => setDraft({ ...draft, serial_port: event.target.value })}>
                      <option value="">Select a port</option>
                      {devices.ports.map((port) => (
                        <option key={port.device} value={port.device}>
                          {port.device} {port.description ? `· ${port.description}` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="warn">Opening the Uno port can reset the board and run its home routine.</p>
                  <div className="row">
                    <button
                      type="button"
                      disabled={busy || active}
                      onClick={() => {
                        if (!window.confirm("Opening the port can reset the Arduino Uno and command its home pose. Connect?")) return;
                        call("/api/serial/connect", { port: draft.serial_port, confirm_reset: true });
                      }}
                    >
                      Connect
                    </button>
                    <button type="button" disabled={busy || active} onClick={() => call("/api/serial/disconnect")}>
                      Disconnect
                    </button>
                  </div>
                  <p className="meta">Planner Ollama is chosen in the command panel.</p>
                  <label>
                    Local VLM endpoint
                    <input value={draft.vlm_base_url} onChange={(event) => setDraft({ ...draft, vlm_base_url: event.target.value })} />
                  </label>
                  <label>
                    VLM model id
                    <input value={draft.vlm_model} onChange={(event) => setDraft({ ...draft, vlm_model: event.target.value })} />
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={draft.vlm_enabled}
                      onChange={(event) => setDraft({ ...draft, vlm_enabled: event.target.checked })}
                    />
                    Use the local VLM on hardware frames
                  </label>
                  <label>
                    E-nose
                    <select value={draft.enose_mode} onChange={(event) => setDraft({ ...draft, enose_mode: event.target.value })}>
                      <option value="mock">Mock</option>
                      <option value="hardware">Connected adapter</option>
                      <option value="unavailable">Unavailable</option>
                    </select>
                  </label>
                  <label>
                    E-nose endpoint
                    <input value={draft.enose_base_url} onChange={(event) => setDraft({ ...draft, enose_base_url: event.target.value })} />
                  </label>
                  <button type="submit" disabled={busy || active}>Save devices</button>
                </form>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
