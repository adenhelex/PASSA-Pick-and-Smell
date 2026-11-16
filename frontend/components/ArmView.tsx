"use client";

import { useEffect, useRef } from "react";
import { JOINTS, JointName, ObservationItem } from "../lib/types";

type V = [number, number, number];
type RGB = [number, number, number];

const SCALE = 0.0118;
const BASE: V = [-2.2, 0, 0.15];
const TABLE_SIDE_MM = 1200;
const LINKS = { base: 70, shoulder: 125, elbow: 125, wrist: 58 };
const ZONES: Record<string, { x: number; y: number }> = {
  left: { x: 170, y: 110 },
  center: { x: 170, y: 0 },
  right: { x: 170, y: -110 },
  front: { x: 245, y: 0 },
  far: { x: 340, y: 0 },
};
const PARK: Record<JointName, number> = {
  base: 90,
  shoulder: 75,
  elbow: 100,
  wrist_vertical: 90,
  wrist_rotation: 90,
  gripper: 10,
};
const DEFAULT_SCENE: ObservationItem[] = [
  { id: "orange_fruit", name: "orange fruit", bbox: [], confidence: 1, uncertain: false, zone: "front", held: false, source: "scene", localization_note: "", color: "orange" },
  { id: "orange_cup", name: "orange cup", bbox: [], confidence: 1, uncertain: false, zone: "right", held: false, source: "scene", localization_note: "", color: "orange" },
  { id: "blue_cup", name: "blue cup", bbox: [], confidence: 1, uncertain: false, zone: "center", held: false, source: "scene", localization_note: "", color: "blue" },
  { id: "green_block", name: "green block", bbox: [], confidence: 1, uncertain: false, zone: "far", held: false, source: "scene", localization_note: "", color: "green" },
];

const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lerp = (a: V, b: V, t: number): V => add(a, mul(sub(b, a), t));
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const css = (rgb: RGB) => `rgb(${rgb.map((v) => Math.round(clamp(v, 0, 255))).join(",")})`;

function unit(a: V): V {
  const n = Math.hypot(a[0], a[1], a[2]);
  return n < 1e-6 ? [0, 1, 0] : [a[0] / n, a[1] / n, a[2] / n];
}

function place(xMm: number, yMm: number, zMm: number): V {
  // Backend +Y is the robot's left. Keep that on +Z so a left turn is a left turn in this view.
  return [BASE[0] + xMm * SCALE, zMm * SCALE, BASE[2] + yMm * SCALE];
}

function armPoints(angles: Record<JointName, number>) {
  const yaw = ((angles.base - 90) * Math.PI) / 180;
  const shoulderPitch = ((angles.shoulder - 90) * Math.PI) / 180;
  const elbowPitch = ((angles.elbow - 90) * Math.PI) / 180;
  const wristPitch = ((angles.wrist_vertical - 90) * Math.PI) / 180;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const at = (radial: number, height: number) => place(radial * c, radial * s, height);
  const x1 = LINKS.shoulder * Math.cos(shoulderPitch);
  const z1 = LINKS.base - LINKS.shoulder * Math.sin(shoulderPitch);
  const x2 = x1 + LINKS.elbow * Math.cos(elbowPitch);
  const z2 = z1 - LINKS.elbow * Math.sin(elbowPitch);
  const x3 = x2 + LINKS.wrist * Math.cos(wristPitch);
  const z3 = z2 - LINKS.wrist * Math.sin(wristPitch);
  const shoulder = at(0, LINKS.base);
  const elbow = at(x1, z1);
  const wrist = at(x2, z2);
  const tip = at(x3, z3);
  const across = unit([-Math.sin(yaw), 0, Math.cos(yaw)]);
  const hand = unit(sub(tip, wrist));
  return { shoulder, elbow, wrist, tip, across, hand, yaw };
}

const BG: RGB = [16, 20, 17];
const FG: RGB = [232, 240, 234];
const neutral = (t: number) => mix(BG, FG, t);
const PALETTE = {
  bench: neutral(0.13),
  grid: neutral(0.28),
  foot: neutral(0.58),
  joint: neutral(0.7),
  metal: neutral(0.42),
  slot: neutral(0.88),
  shell: mix(neutral(0.35), [68, 126, 188], 0.74),
  accent: mix(neutral(0.35), [68, 126, 188], 0.94),
  nose: mix(neutral(0.35), [38, 146, 125], 0.7),
  signal: mix(neutral(0.35), [38, 146, 125], 0.98),
  light: neutral(0.03),
};
const NAMED: Record<string, RGB> = {
  orange: [212, 134, 55],
  blue: [90, 150, 210],
  green: [70, 170, 90],
  red: [200, 70, 60],
  purple: [150, 110, 190],
  yellow: [210, 180, 70],
  white: [236, 236, 232],
  gray: [148, 150, 148],
  grey: [148, 150, 148],
  black: [46, 48, 50],
  clear: [206, 222, 228],
  brown: [142, 96, 58],
  pink: [214, 140, 160],
};

function objectKind(name: string) {
  const text = name.toLowerCase();
  if (text.includes("mouse")) return "mouse";
  if (text.includes("phone")) return "phone";
  if (text.includes("bottle")) return "bottle";
  if (text.includes("cup") || text.includes("glass")) return "cup";
  if (text.includes("block")) return "block";
  if (text.includes("orange") || text.includes("fruit") || text.includes("apple")) return "orange";
  return "block";
}

function tableTwin(item: ObservationItem): { at: V; wide: number; tall: number } | null {
  const zone = item.zone ? ZONES[item.zone] : undefined;
  if (zone) return { at: place(zone.x, zone.y, 0), wide: 42, tall: 70 };
  if (item.bbox.length < 4) return null;
  const [u, v, w, h] = item.bbox;
  if ((w * h) / (640 * 480) > 0.2) return null;
  const across = clamp((u + w / 2) / 640, 0, 1);
  const depth = 1 - clamp((v + h / 2) / 480, 0, 1);
  const forward = 80 + depth * (TABLE_SIDE_MM - 160);
  const lateral = (across - 0.5) * (TABLE_SIDE_MM - 80);
  return {
    at: place(forward, lateral, 0),
    wide: clamp((w / 640) * 700, 40, 130),
    tall: clamp((h / 480) * 300, 30, 180),
  };
}

interface Scene {
  angles: Record<JointName, number | null>;
  jointsKnown: boolean;
  items: ObservationItem[];
  selectedId: string | null;
  phase: string;
  showDefaultScene: boolean;
  yaw: number;
  pitch: number;
  dragging: boolean;
}

export default function ArmView(props: Omit<Scene, "yaw" | "pitch" | "dragging">) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scene = useRef<Scene>({ ...props, yaw: -Math.PI / 2 + 0.38, pitch: 0.5, dragging: false });
  scene.current = { ...scene.current, ...props };
  const kick = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let running = false;
    const shown: Record<JointName, number> = { ...PARK };
    const pointer = { x: 0, y: 0, active: false };

    const frame = () => {
      let moving = false;
      const goal = scene.current.jointsKnown ? scene.current.angles : null;
      if (goal) {
        for (const { key } of JOINTS) {
          const target = goal[key];
          if (target == null) continue;
          if (shown[key] !== target) {
            shown[key] = target;
            moving = true;
          }
        }
      }
      paint(ctx, canvas, shown, scene.current);
      if (moving || scene.current.dragging) raf = requestAnimationFrame(frame);
      else running = false;
    };
    const start = () => {
      if (running) return;
      running = true;
      raf = requestAnimationFrame(frame);
    };
    kick.current = start;

    const down = (event: PointerEvent) => {
      pointer.active = true;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      scene.current.dragging = true;
      canvas.setPointerCapture(event.pointerId);
      start();
    };
    const move = (event: PointerEvent) => {
      if (!pointer.active) return;
      scene.current.yaw -= (event.clientX - pointer.x) * 0.008;
      scene.current.pitch = clamp(scene.current.pitch + (event.clientY - pointer.y) * 0.006, 0.18, 1.15);
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      start();
    };
    const up = () => {
      pointer.active = false;
      scene.current.dragging = false;
    };
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    const observer = new ResizeObserver(() => start());
    observer.observe(canvas);
    start();
    return () => {
      cancelAnimationFrame(raf);
      running = false;
      observer.disconnect();
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
    };
  }, []);

  useEffect(() => {
    kick.current();
  }, [props.angles, props.jointsKnown, props.items, props.selectedId, props.phase, props.showDefaultScene]);

  return (
    <div className="arm-view">
      <canvas ref={canvasRef} aria-label="3D view of the arm and the objects on the table. Drag to orbit." />
      <button
        type="button"
        className="arm-orbit"
        onClick={() => {
          scene.current.yaw += Math.PI / 6;
          kick.current();
        }}
      >
        Rotate view
      </button>
      {!props.jointsKnown && <p className="arm-note">Joint angles are unknown until the next status sync.</p>}
    </div>
  );
}

function paint(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  angles: Record<JointName, number>,
  scene: Scene,
) {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  if (width < 2 || height < 2) return;
  const ratio = window.devicePixelRatio || 1;
  const nextWidth = Math.floor(width * ratio);
  const nextHeight = Math.floor(height * ratio);
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const tableSide = TABLE_SIDE_MM * SCALE;
  const focus: V = [BASE[0] + tableSide * 0.36, 0.45, BASE[2]];
  const distance = width < 520 ? 22 : 17;
  const eye = add(focus, [
    Math.sin(scene.yaw) * Math.cos(scene.pitch) * distance,
    Math.sin(scene.pitch) * distance,
    Math.cos(scene.yaw) * Math.cos(scene.pitch) * distance,
  ]);
  const forward = unit(sub(focus, eye));
  const right = unit(cross(forward, [0, 1, 0]));
  const vertical = cross(right, forward);
  const basis = { eye, forward, right, vertical };
  const focal = Math.min(width * 1.2, height * 1.5);
  const offsetY = height * 0.52;
  const light = unit([-0.45, 0.87, 0.32]);
  const faces: Array<{ screen: V[]; fill: string; z: number }> = [];

  const project = (point: V): V | null => {
    const delta = sub(point, basis.eye);
    const depth = dot(delta, basis.forward);
    if (depth < 0.055) return null;
    const scale = focal / depth;
    return [width / 2 + dot(delta, basis.right) * scale, offsetY - dot(delta, basis.vertical) * scale, depth];
  };
  const depthOf = (point: V) => dot(sub(point, basis.eye), basis.forward);
  const clipNear = (points: V[]) => {
    const clipped: V[] = [];
    let previous = points[points.length - 1];
    let previousDepth = depthOf(previous);
    for (const point of points) {
      const nextDepth = depthOf(point);
      if ((previousDepth >= 0.055) !== (nextDepth >= 0.055)) {
        clipped.push(lerp(previous, point, (0.055 - previousDepth) / (nextDepth - previousDepth)));
      }
      if (nextDepth >= 0.055) clipped.push(point);
      previous = point;
      previousDepth = nextDepth;
    }
    return clipped;
  };
  const trace = (points: Array<V | null>) => {
    const visible = points.filter((point): point is V => point !== null);
    if (visible.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(visible[0][0], visible[0][1]);
    visible.slice(1).forEach((point) => ctx.lineTo(point[0], point[1]));
  };
  const surface = (points: V[], center: V, material: RGB) => {
    const normal = unit(cross(sub(points[1], points[0]), sub(points[2], points[0])));
    const outward = dot(normal, sub(points[0], center)) < 0 ? mul(normal, -1) : normal;
    if (dot(outward, sub(basis.eye, points[0])) <= 0) return;
    const screen = clipNear(points).map(project);
    if (screen.some((point) => point === null) || screen.length < 3) return;
    const drawn = screen as V[];
    const shade = 0.09 + 0.2 * dot(outward, light);
    const shaded = shade < 0 ? mix(material, PALETTE.slot, -shade) : mix(material, PALETTE.light, shade);
    faces.push({
      screen: drawn,
      fill: css(shaded),
      z: drawn.reduce((sum, point) => sum + point[2], 0) / drawn.length,
    });
  };
  const drawFaces = () => {
    faces.sort((a, b) => b.z - a.z);
    for (const face of faces) {
      trace(face.screen);
      ctx.closePath();
      ctx.fillStyle = face.fill;
      ctx.strokeStyle = face.fill;
      ctx.lineWidth = 0.6;
      ctx.fill();
      ctx.stroke();
    }
    faces.length = 0;
  };
  const ring = (center: V, a: V, b: V, radius: number, count: number) =>
    Array.from({ length: count }, (_, index) => {
      const angle = (2 * Math.PI * index) / count;
      return add(center, add(mul(a, radius * Math.cos(angle)), mul(b, radius * Math.sin(angle))));
    });
  const tube = (start: V, end: V, r1: number, r2: number, material: RGB, count = 16) => {
    if (Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]) < 1e-4) return;
    const center = lerp(start, end, 0.5);
    const axis = unit(sub(end, start));
    const a = unit(cross(axis, Math.abs(axis[1]) > 0.94 ? [1, 0, 0] : [0, 1, 0]));
    const b = cross(axis, a);
    const low = ring(start, a, b, r1, count);
    const high = ring(end, a, b, r2, count);
    for (let index = 0; index < count; index += 1) {
      const next = (index + 1) % count;
      surface([low[index], low[next], high[next], high[index]], center, material);
    }
    surface(low, center, material);
    surface(high, center, material);
  };
  const orientedBox = (center: V, size: V, axes: [V, V, V], material: RGB) => {
    const half = size.map((value) => value / 2) as V;
    const signs: Array<[number, number, number]> = [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ];
    const corners = signs.map(([sx, sy, sz]) =>
      add(center, add(add(mul(axes[0], sx * half[0]), mul(axes[1], sy * half[1])), mul(axes[2], sz * half[2]))),
    );
    for (const face of [
      [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2], [0, 1, 5, 4], [3, 2, 6, 7],
    ]) {
      surface(face.map((index) => corners[index]), center, material);
    }
  };
  const sphere = (center: V, radius: number, material: RGB) => {
    const count = 14;
    const bands = 8;
    const rows = Array.from({ length: bands + 1 }, (_, index) => {
      const angle = (index * Math.PI) / bands;
      return ring(add(center, [0, Math.cos(angle) * radius, 0]), [1, 0, 0], [0, 0, 1], Math.max(0.0001, Math.sin(angle) * radius), count);
    });
    for (let band = 0; band < bands; band += 1) {
      for (let index = 0; index < count; index += 1) {
        const next = (index + 1) % count;
        surface([rows[band][index], rows[band][next], rows[band + 1][next], rows[band + 1][index]], center, material);
      }
    }
  };
  const strokeLine = (a: V, b: V, color: string, weight: number) => {
    const da = depthOf(a);
    const db = depthOf(b);
    if (Math.max(da, db) < 0.055) return;
    const start = da < 0.055 ? lerp(a, b, (0.055 - da) / (db - da)) : a;
    const end = db < 0.055 ? lerp(b, a, (0.055 - db) / (da - db)) : b;
    trace([project(start), project(end)]);
    ctx.strokeStyle = color;
    ctx.lineWidth = weight;
    ctx.stroke();
  };

  const nearX = BASE[0];
  const farX = BASE[0] + tableSide;
  const leftZ = BASE[2] - tableSide / 2;
  const rightZ = BASE[2] + tableSide / 2;
  orientedBox([(nearX + farX) / 2, -0.14, BASE[2]], [tableSide, 0.28, tableSide], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], PALETTE.bench);
  drawFaces();
  const grid = 100 * SCALE;
  for (let x = nearX; x <= farX + 0.01; x += grid) strokeLine([x, 0.008, leftZ], [x, 0.008, rightZ], css(PALETTE.grid), 0.6);
  for (let z = leftZ; z <= rightZ + 0.01; z += grid) strokeLine([nearX, 0.008, z], [farX, 0.008, z], css(PALETTE.grid), 0.6);

  const pose = armPoints(angles);
  const items = scene.items.length ? scene.items : scene.showDefaultScene ? DEFAULT_SCENE : [];
  const drawn = items.flatMap((item) => {
    const twin = tableTwin(item);
    if (!twin) return [];
    if (item.held) {
      return [{ item, at: add(pose.tip, mul(pose.hand, 0.12)), wide: twin.wide, tall: twin.tall, lifted: true }];
    }
    return [{ item, ...twin, lifted: false }];
  });

  for (const { item, at, wide } of drawn) {
    if (item.held) continue;
    const radius = Math.max(0.22, wide * SCALE * 0.62);
    const marker = ring([at[0], 0.02, at[2]], [1, 0, 0], [0, 0, 1], radius, 28);
    const color = item.id === scene.selectedId ? css(PALETTE.accent) : css(PALETTE.grid);
    const weight = item.id === scene.selectedId ? 1.6 : 0.8;
    for (let index = 0; index < marker.length; index += 1) {
      strokeLine(marker[index], marker[(index + 1) % marker.length], color, weight);
    }
  }

  const up: V = [0, 1, 0];
  const column = (y: number): V => [BASE[0], y, BASE[2]];
  tube(column(0.02), column(0.16), 0.46, 0.46, PALETTE.foot, 20);
  tube(column(0.16), column(0.42), 0.36, 0.34, PALETTE.joint, 18);
  tube(column(0.42), column(0.62), 0.34, 0.24, PALETTE.shell, 18);
  tube(column(0.62), pose.shoulder, 0.22, 0.22, PALETTE.shell, 16);
  const link = (a: V, b: V, radius: number) => {
    tube(a, b, radius, radius * 0.84, PALETTE.shell);
    const strip = mul(pose.across, radius * 0.8);
    tube(add(lerp(a, b, 0.18), strip), add(lerp(a, b, 0.82), strip), 0.045, 0.045, PALETTE.metal, 8);
  };
  link(pose.shoulder, pose.elbow, 0.2);
  link(pose.elbow, pose.wrist, 0.16);
  link(pose.wrist, pose.tip, 0.11);
  for (const [point, radius, span] of [
    [pose.shoulder, 0.28, 0.24],
    [pose.elbow, 0.24, 0.2],
    [pose.wrist, 0.18, 0.14],
  ] as Array<[V, number, number]>) {
    tube(add(point, mul(pose.across, -span)), add(point, mul(pose.across, span)), radius, radius, PALETTE.joint, 16);
  }
  const inlet = add(pose.wrist, add(mul(pose.across, 0.22), [0, 0.08, 0]));
  const hose = [
    column(0.4),
    add(pose.shoulder, mul(pose.across, 0.26)),
    add(lerp(pose.shoulder, pose.elbow, 0.5), mul(pose.across, 0.2)),
    add(pose.elbow, mul(pose.across, 0.22)),
    add(lerp(pose.elbow, pose.wrist, 0.55), mul(pose.across, 0.16)),
    inlet,
  ];
  for (let index = 1; index < hose.length; index += 1) tube(hose[index - 1], hose[index], 0.032, 0.032, PALETTE.signal, 8);
  tube(inlet, add(inlet, mul(pose.hand, 0.16)), 0.055, 0.04, PALETTE.nose, 12);
  const camera = add(pose.wrist, add(mul(pose.across, -0.18), [0, 0.12, 0]));
  orientedBox(camera, [0.22, 0.14, 0.18], [pose.across, up, pose.hand], PALETTE.accent);

  const roll = ((angles.wrist_rotation - 90) * Math.PI) / 180;
  const side0 = unit(cross(pose.hand, Math.abs(pose.hand[1]) > 0.92 ? pose.across : up));
  const binormal = unit(cross(side0, pose.hand));
  const fingerSide = unit(add(mul(side0, Math.cos(roll)), mul(binormal, Math.sin(roll))));
  const fingerUp = unit(cross(pose.hand, fingerSide));
  const gap = 0.16 - clamp((angles.gripper - 10) / 63, 0, 1) * 0.11;
  for (const sign of [-1, 1]) {
    const pad = add(pose.tip, add(mul(fingerSide, sign * gap), mul(fingerUp, -0.02)));
    orientedBox(pad, [0.07, 0.28, 0.09], [fingerSide, fingerUp, pose.hand], PALETTE.metal);
  }

  const labels: Array<{ at: V; name: string; selected: boolean }> = [];
  for (const { item, at, wide, tall, lifted } of drawn) {
    const color = NAMED[item.color] ?? NAMED.gray;
    const kind = objectKind(item.name);
    const span = wide * SCALE;
    const rise = tall * SCALE;
    const ground: V = lifted ? at : [at[0], 0, at[2]];
    const top = (height: number) => add(ground, [0, lifted ? height / 2 : height, 0]);
    let crown = rise;
    if (kind === "orange") {
      const radius = Math.max(span, rise) * 0.42;
      sphere(add(ground, [0, lifted ? 0 : radius, 0]), radius, color);
      crown = radius * (lifted ? 1 : 2);
    } else if (kind === "cup" || kind === "bottle") {
      const height = kind === "bottle" ? Math.max(rise, span * 1.7) : Math.max(rise, span * 0.9);
      const radius = span * (kind === "bottle" ? 0.28 : 0.42);
      const foot = lifted ? add(ground, [0, -height / 2, 0]) : ground;
      const lip = add(foot, [0, height, 0]);
      tube(foot, lip, radius * 0.82, radius, color, 16);
      tube(lip, add(lip, [0, 0.035, 0]), radius * 0.9, radius * 0.9, PALETTE.slot, 16);
      crown = height + 0.04;
    } else if (kind === "mouse") {
      const height = Math.max(0.08, Math.min(rise, span * 0.38));
      orientedBox(add(ground, [0, lifted ? 0 : height / 2, 0]), [span, height, span * 0.62], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], color);
      crown = height;
    } else if (kind === "phone") {
      const height = Math.max(0.04, span * 0.12);
      orientedBox(add(ground, [0, lifted ? 0 : height / 2, 0]), [span * 0.55, height, span], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], color);
      crown = height;
    } else {
      const height = Math.max(rise, span * 0.7);
      orientedBox(add(ground, [0, lifted ? 0 : height / 2, 0]), [span, height, span * 0.8], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], color);
      crown = height;
    }
    labels.push({ at: add(top(crown), [0, 0.12, 0]), name: item.name, selected: item.id === scene.selectedId });
  }
  drawFaces();
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  for (const label of labels) {
    const screen = project(label.at);
    if (!screen) continue;
    ctx.fillStyle = label.selected ? css(PALETTE.accent) : css(neutral(0.78));
    ctx.fillText(label.name, screen[0], screen[1]);
  }

  if (scene.phase === "sampling") {
    const held = drawn.find((entry) => entry.item.held);
    if (held) {
      for (let index = 0; index < 3; index += 1) {
        const from = add(held.at, [0, 0.2 + index * 0.04, 0]);
        strokeLine(lerp(from, inlet, 0.15), lerp(from, inlet, 0.85), css(PALETTE.signal), 1.6);
      }
    }
  }
}
