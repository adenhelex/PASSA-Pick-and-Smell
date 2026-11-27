"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { JOINTS, JointName, ObservationItem } from "../lib/types";

const SCALE = 0.0118;
const BASE: [number, number, number] = [-2.2, 0, 0.15];
const TABLE_SIDE_MM = 560;
const MODEL_ROOT = "/models/braccio/stl";
/** Braccio.Begin() / config poses.park — C-shaped rest from the reference photo. */
const PARK: Record<JointName, number> = {
  base: 90,
  shoulder: 45,
  elbow: 180,
  wrist_vertical: 180,
  wrist_rotation: 90,
  cylinder: 73,
};
const ZONES: Record<string, { x: number; y: number }> = {
  left: { x: 170, y: 110 },
  center: { x: 170, y: 0 },
  right: { x: 170, y: -110 },
  front: { x: 245, y: 0 },
  far: { x: 340, y: 0 },
};
const DEFAULT_SCENE: ObservationItem[] = [
  { id: "orange_fruit", name: "orange fruit", bbox: [], confidence: 1, uncertain: false, zone: "front", held: false, source: "scene", localization_note: "", color: "orange" },
  { id: "orange_cup", name: "orange cup", bbox: [], confidence: 1, uncertain: false, zone: "right", held: false, source: "scene", localization_note: "", color: "orange" },
  { id: "blue_cup", name: "blue cup", bbox: [], confidence: 1, uncertain: false, zone: "center", held: false, source: "scene", localization_note: "", color: "blue" },
  { id: "green_block", name: "green block", bbox: [], confidence: 1, uncertain: false, zone: "far", held: false, source: "scene", localization_note: "", color: "green" },
];
const NAMED: Record<string, number> = {
  orange: 0xd48637,
  blue: 0x5a96d2,
  green: 0x46aa5a,
  red: 0xc8463c,
  purple: 0x966ebe,
  yellow: 0xd2b446,
  white: 0xecece8,
  gray: 0x949694,
  grey: 0x949694,
  black: 0x2e3032,
  clear: 0xcedee4,
  brown: 0x8e603a,
  pink: 0xd68ca0,
};

type ArmParts = {
  base: THREE.Object3D;
  shoulder: THREE.Object3D;
  elbow: THREE.Object3D;
  wristPitch: THREE.Object3D;
  wristRoll: THREE.Object3D;
  coverPivot: THREE.Object3D;
  tip: THREE.Object3D;
};

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function place(xMm: number, yMm: number, zMm: number): THREE.Vector3 {
  return new THREE.Vector3(BASE[0] + xMm * SCALE, zMm * SCALE, BASE[2] + yMm * SCALE);
}

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

function tableTwin(item: ObservationItem) {
  if (item.bbox.length >= 4) {
    const [u, v, w, h] = item.bbox;
    if ((w * h) / (640 * 480) <= 0.92) {
      const across = clamp((u + w / 2) / 640, 0, 1);
      // The foot of the box is where it stands. The bottom of the photo is the
      // base, and the top is the far edge of the table.
      const foot = clamp((v + h) / 480, 0, 1);
      const forward = (1 - foot) * TABLE_SIDE_MM;
      // Image-left is the low base end and sits on −Y. Image-right sits on +Y.
      const lateral = (across - 0.5) * TABLE_SIDE_MM;
      const kind = objectKind(item.name);
      const tallish = kind === "bottle" || h > w * 1.2;
      return {
        at: place(forward, lateral, 0),
        wide: clamp((w / 640) * 420, tallish ? 90 : 120, tallish ? 140 : 200),
        tall: clamp((h / 480) * 420, tallish ? 200 : 150, tallish ? 280 : 240),
      };
    }
  }
  const zone = item.zone ? ZONES[item.zone] : undefined;
  if (zone) return { at: place(zone.x, zone.y, 0), wide: 150, tall: 230 };
  return null;
}

function rad(deg: number) {
  return (deg * Math.PI) / 180;
}

/** ros_braccio_urdf servo mapping — Begin() looks like the reference photo. */
function applyBraccioPose(
  parts: ArmParts,
  angles: Record<JointName, number>,
  cover: { park: number; sample: number; parkTilt?: number; sampleTilt?: number } = { park: 73, sample: 10 },
) {
  // Base 90 faces straight onto the table. A lower angle turns toward image-left.
  parts.base.rotation.z = -rad(angles.base);
  parts.shoulder.rotation.x = rad(angles.shoulder);
  parts.elbow.rotation.x = rad(angles.elbow);
  parts.wristPitch.rotation.x = rad(angles.wrist_vertical);
  parts.wristRoll.rotation.z = -rad(angles.wrist_rotation);
  // Dry-run map, same numbers as config/default.yaml. Sample servo levels the
  // rim on the table. Park servo holds the cup sideways. Not calibrated.
  const span = cover.sample - cover.park;
  const share = span === 0 ? 0 : (angles.cylinder - cover.park) / span;
  const parkTilt = cover.parkTilt ?? -63;
  const sampleTilt = cover.sampleTilt ?? -38;
  const tilt = parkTilt + share * (sampleTilt - parkTilt);
  parts.coverPivot.rotation.y = rad(tilt);
}

async function loadMesh(loader: STLLoader, file: string, color: number) {
  const geometry = await loader.loadAsync(`${MODEL_ROOT}/${file}`);
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.18,
    roughness: 0.55,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function makeVisual(mesh: THREE.Object3D, xyz: [number, number, number], rpy: [number, number, number]) {
  const visual = new THREE.Group();
  visual.position.set(...xyz);
  visual.rotation.set(...rpy);
  visual.add(mesh);
  return visual;
}

function jointOrigin(parent: THREE.Object3D, xyz: [number, number, number], rpy: [number, number, number]) {
  const origin = new THREE.Group();
  origin.position.set(...xyz);
  origin.rotation.set(...rpy);
  parent.add(origin);
  const joint = new THREE.Group();
  origin.add(joint);
  return joint;
}

async function loadCoverMesh(loader: STLLoader, file: string, part: "cylinder" | "fixed") {
  // These files are already in the wrist-mesh design frame, taken from
  // PASSA Cylindrical Sampling Cover.stl. The wrist triangles stay on the
  // arm mesh. The cylinder, including its sampling tube, is the rotating part.
  const geometry = await loader.loadAsync(`${MODEL_ROOT}/${file}`);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial(
      part === "cylinder"
        ? { color: 0xd5e6ee, metalness: 0.04, roughness: 0.32, transparent: true, opacity: 0.48, side: THREE.DoubleSide, depthWrite: false }
        : { color: 0xcfcfcf, metalness: 0.18, roughness: 0.55 },
    ),
  );
  mesh.scale.setScalar(0.001);
  return mesh;
}

async function buildBraccioArm(loader: STLLoader): Promise<{ root: THREE.Group; parts: ArmParts }> {
  const orange = 0x913600;
  const white = 0xcfcfcf;
  const [baseMesh, shoulderMesh, elbowMesh, wristPitchMesh, wristRollMesh, shroud, bracket] = await Promise.all([
    loadMesh(loader, "braccio_base.stl", orange),
    loadMesh(loader, "braccio_shoulder.stl", orange),
    loadMesh(loader, "braccio_elbow.stl", orange),
    loadMesh(loader, "braccio_wrist_pitch.stl", orange),
    loadMesh(loader, "braccio_wrist_roll.stl", white),
    loadCoverMesh(loader, "cover-cylinder.stl", "cylinder"),
    loadCoverMesh(loader, "cover-fixed.stl", "fixed"),
  ]);

  // URDF meshes are mm → meters; mount scale maps into the table scene.
  const mm = 0.001;
  for (const mesh of [baseMesh, shoulderMesh, elbowMesh, wristPitchMesh, wristRollMesh]) {
    mesh.scale.setScalar(mm);
  }

  const root = new THREE.Group();
  const base = jointOrigin(root, [0, 0, 0], [0, 0, 0]);
  base.add(makeVisual(baseMesh, [0, 0.004, 0], [0, 0, Math.PI]));

  const shoulder = jointOrigin(base, [0, -0.002, 0.072], [-Math.PI / 2, 0, 0]);
  shoulder.add(makeVisual(shoulderMesh, [-0.0045, 0.0055, -0.026], [0, 0, 0]));

  const elbow = jointOrigin(shoulder, [0, 0, 0.125], [-Math.PI / 2, 0, 0]);
  elbow.add(makeVisual(elbowMesh, [-0.0045, 0.005, -0.025], [0, 0, 0]));

  const wristPitch = jointOrigin(elbow, [0, 0, 0.125], [-Math.PI / 2, 0, 0]);
  wristPitch.add(makeVisual(wristPitchMesh, [0.003, -0.0004, -0.024], [0, 0, 0]));

  const wristRoll = jointOrigin(wristPitch, [0, 0, 0.06], [0, 0, 0]);
  wristRoll.add(makeVisual(wristRollMesh, [0.006, 0, 0], [0, 0, 0]));

  // Same visual origin as the wrist mesh. Vertices are already design millimetres.
  const toolMount = new THREE.Group();
  toolMount.position.set(0.006, 0, 0);
  wristRoll.add(toolMount);
  toolMount.add(bracket);
  const pivot = new THREE.Vector3(-6 * mm, 40.5 * mm, 47 * mm);
  const coverPivot = new THREE.Group();
  coverPivot.position.copy(pivot);
  toolMount.add(coverPivot);
  shroud.position.set(-pivot.x, -pivot.y, -pivot.z);
  coverPivot.add(shroud);

  const tip = new THREE.Object3D();
  tip.position.set(0, 0, 0.08);
  wristRoll.add(tip);

  return {
    root,
    parts: { base, shoulder, elbow, wristPitch, wristRoll, coverPivot, tip },
  };
}

function makeTable() {
  const side = TABLE_SIDE_MM * SCALE;
  const group = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(side, 0.28, side),
    new THREE.MeshStandardMaterial({ color: 0x2a322c, roughness: 0.9, metalness: 0.05 }),
  );
  top.position.set(BASE[0] + side / 2, -0.14, BASE[2]);
  top.receiveShadow = true;
  group.add(top);

  const grid = new THREE.GridHelper(side, 12, 0x4a584e, 0x3a453f);
  grid.position.set(BASE[0] + side / 2, 0.01, BASE[2]);
  group.add(grid);
  return group;
}

function makeObjectMesh(item: ObservationItem, wide: number, tall: number) {
  const kind = objectKind(item.name);
  const color = NAMED[item.color] ?? NAMED.gray;
  const span = wide * SCALE;
  const rise = tall * SCALE;
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.08 });
  if (kind === "orange") {
    const radius = Math.max(span, rise) * 0.5;
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 16), material);
    mesh.position.y = radius;
    return mesh;
  }
  if (kind === "cup" || kind === "bottle") {
    const height = kind === "bottle" ? Math.max(rise, span * 1.8) : Math.max(rise, span * 1.15);
    const radius = span * (kind === "bottle" ? 0.34 : 0.48);
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.82, height, 18), material);
    mesh.position.y = height / 2;
    return mesh;
  }
  if (kind === "mouse") {
    const height = Math.max(0.12, Math.min(rise, span * 0.4));
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(span, height, span * 0.62), material);
    mesh.position.y = height / 2;
    return mesh;
  }
  const height = Math.max(rise, span * 0.85);
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(span, height, span * 0.85), material);
  mesh.position.y = height / 2;
  return mesh;
}

function prepareSafariWebGL() {
  const prototype = WebGL2RenderingContext.prototype as WebGL2RenderingContext & { __passaWebGL?: boolean };
  if (prototype.__passaWebGL) return;
  const readPrecision = prototype.getShaderPrecisionFormat;
  prototype.getShaderPrecisionFormat = function (shaderType: number, precisionType: number) {
    const format = readPrecision.call(this, shaderType, precisionType);
    if (format) return format;
    const high = precisionType === this.HIGH_FLOAT || precisionType === this.HIGH_INT;
    return { rangeMin: high ? 127 : 15, rangeMax: high ? 127 : 15, precision: high ? 23 : 10 };
  };
  const readParameter = prototype.getParameter;
  prototype.getParameter = function (pname: number) {
    const value = readParameter.call(this, pname);
    if (value != null) return value;
    if (pname === this.VERSION) return "WebGL 2.0";
    if (pname === this.SHADING_LANGUAGE_VERSION) return "WebGL GLSL ES 3.00";
    if (pname === this.VENDOR) return "WebKit";
    if (pname === this.RENDERER) return "WebKit WebGL";
    if (pname === this.SCISSOR_BOX || pname === this.VIEWPORT) {
      return new Int32Array([0, 0, this.canvas.width || 300, this.canvas.height || 150]);
    }
    return value;
  };
  prototype.__passaWebGL = true;
}

interface SceneProps {
  angles: Record<JointName, number | null>;
  jointsKnown: boolean;
  items: ObservationItem[];
  selectedId: string | null;
  phase: string;
  showDefaultScene: boolean;
  cover?: { park: number; sample: number; parkTilt?: number; sampleTilt?: number };
}

export default function ArmView(props: SceneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const propsRef = useRef(props);
  const bumpYaw = useRef<() => void>(() => {});
  propsRef.current = props;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101411);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 80);
    prepareSafariWebGL();
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = false;
    host.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight(0xdde8df, 0x1a201c, 1.1);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.15);
    key.position.set(-6, 10, 4);
    key.castShadow = false;
    scene.add(key);

    scene.add(makeTable());

    const objects = new THREE.Group();
    scene.add(objects);

    const orbit = {
      yaw: -Math.PI / 2 + 0.38,
      pitch: 0.5,
      dragging: false,
      x: 0,
      y: 0,
    };
    const shown: Record<JointName, number> = { ...PARK };
    let parts: ArmParts | null = null;
    let armRoot: THREE.Group | null = null;
    let raf = 0;
    let alive = true;
    let status = "Loading Braccio CAD meshes…";

    const label = document.createElement("div");
    label.className = "arm-note";
    label.textContent = status;
    host.appendChild(label);

    const frame = () => {
      if (!alive) return;
      const current = propsRef.current;
      // Always drive the twin: park (Begin) until joints sync, then live angles.
      if (parts) {
        if (current.jointsKnown) {
          for (const { key: joint } of JOINTS) {
            const target = current.angles[joint];
            if (target == null) continue;
            shown[joint] = target;
          }
        }
        applyBraccioPose(parts, shown, current.cover);
      }

      const tableSide = TABLE_SIDE_MM * SCALE;
      const focus = new THREE.Vector3(BASE[0] + tableSide * 0.36, 0.45, BASE[2]);
      const distance = host.clientWidth < 520 ? 22 : 17;
      camera.position.set(
        focus.x + Math.sin(orbit.yaw) * Math.cos(orbit.pitch) * distance,
        focus.y + Math.sin(orbit.pitch) * distance,
        focus.z + Math.cos(orbit.yaw) * Math.cos(orbit.pitch) * distance,
      );
      camera.lookAt(focus);

      for (const child of [...objects.children]) {
        child.traverse((node) => {
          const mesh = node as THREE.Mesh;
          if (mesh.geometry) mesh.geometry.dispose();
          const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
          if (Array.isArray(mat)) mat.forEach((entry) => entry.dispose());
          else mat?.dispose?.();
        });
        objects.remove(child);
      }

      const items = current.items.length ? current.items : current.showDefaultScene ? DEFAULT_SCENE : [];
      for (const item of items) {
        const twin = tableTwin(item);
        if (!twin) continue;
        const mesh = makeObjectMesh(item, twin.wide, twin.tall);
        if (item.held && parts) {
          const tip = new THREE.Vector3();
          parts.tip.getWorldPosition(tip);
          mesh.position.copy(tip);
        } else {
          mesh.position.copy(twin.at);
        }
        if (item.id === current.selectedId) {
          const ring = new THREE.Mesh(
            new THREE.RingGeometry(Math.max(0.22, twin.wide * SCALE * 0.7), Math.max(0.28, twin.wide * SCALE * 0.9), 28),
            new THREE.MeshBasicMaterial({ color: 0x8eb6c9, side: THREE.DoubleSide }),
          );
          ring.rotation.x = -Math.PI / 2;
          ring.position.set(mesh.position.x, 0.02, mesh.position.z);
          objects.add(ring);
        }
        objects.add(mesh);
      }

      renderer.render(scene, camera);
      raf = requestAnimationFrame(frame);
    };

    const resize = () => {
      const width = Math.max(host.clientWidth, 2);
      const height = Math.max(host.clientHeight, 2);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(host);

    const down = (event: PointerEvent) => {
      orbit.dragging = true;
      orbit.x = event.clientX;
      orbit.y = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
    };
    const move = (event: PointerEvent) => {
      if (!orbit.dragging) return;
      orbit.yaw -= (event.clientX - orbit.x) * 0.008;
      orbit.pitch = clamp(orbit.pitch + (event.clientY - orbit.y) * 0.006, 0.18, 1.15);
      orbit.x = event.clientX;
      orbit.y = event.clientY;
    };
    const up = () => {
      orbit.dragging = false;
    };
    renderer.domElement.addEventListener("pointerdown", down);
    renderer.domElement.addEventListener("pointermove", move);
    renderer.domElement.addEventListener("pointerup", up);
    renderer.domElement.addEventListener("pointercancel", up);
    bumpYaw.current = () => {
      orbit.yaw += Math.PI / 6;
    };

    const loader = new STLLoader();
    buildBraccioArm(loader)
      .then(({ root, parts: next }) => {
        if (!alive) return;
        // URDF Z-up → scene Y-up, yaw so Begin stands on the table and curls over +X.
        const mount = new THREE.Group();
        mount.position.set(BASE[0], 0, BASE[2]);
        mount.scale.setScalar(1000 * SCALE);
        const zUp = new THREE.Group();
        zUp.rotation.order = "XYZ";
        zUp.rotation.set(Math.PI / 2, Math.PI, 0);
        mount.add(zUp);
        zUp.add(root);
        armRoot = mount;
        parts = next;
        applyBraccioPose(parts, PARK);
        scene.add(mount);
        status = "Cylinder sampling cover. Fit and servo alignment are not verified.";
        label.textContent = status;
        label.style.display = "";
      })
      .catch((error) => {
        status = error instanceof Error ? error.message : "Could not load Braccio meshes.";
        label.textContent = status;
        label.style.display = "";
      });

    raf = requestAnimationFrame(frame);

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", down);
      renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", up);
      renderer.domElement.removeEventListener("pointercancel", up);
      if (armRoot) scene.remove(armRoot);
      renderer.dispose();
      host.removeChild(renderer.domElement);
      host.removeChild(label);
    };
  }, []);

  return (
    <div className="arm-view" ref={hostRef}>
      <button type="button" className="arm-orbit" onClick={() => bumpYaw.current()}>
        Rotate view
      </button>
      {!props.jointsKnown && <p className="arm-note">Joint angles are unknown until the next status sync.</p>}
    </div>
  );
}
