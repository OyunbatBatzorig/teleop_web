// Quest 3 web teleop: episode interface (v2, 2026-11-10: 60 % see-through robot and ghosts, controller models, scaled ghosts, thumb-rest anchor; force / width sliders at the anchor).
// Layout (user's sketch): digital G1 in the middle, point cloud in front of it, head camera panel above,
// see-through ghost grippers on the controllers, wrist camera panels only while the grip button is held, following the ghost.
// The operator stands where the robot's head camera is (first person); both sticks pressed moves / turns / scales the
// whole scene, so the robot can also be looked at from outside.
import * as THREE from 'three';
const APP_BUILD = 'build 2026-11-22e';   // shown in the HUD: tells which page version the headset runs
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';

// ---------- robot constants (same as xr_teleoperate ghost_vr.py / ghost_overlay.py)
const URDF_URL = '/robot/g1_29dof_mode_15_with_dex1_1.urdf';
const SIDES = ['left', 'right'];
const ARM_JOINTS = SIDES.flatMap(s => ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll',
  'wrist_pitch', 'wrist_yaw'].map(j => `${s}_${j}_joint`));
const LEG_JOINTS = SIDES.flatMap(s => ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
  .map(j => `${s}_${j}_joint`));
const WAIST_JOINTS = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'];
const FINGER_OPEN = 0.0245, FINGER_CLOSED = -0.02;   // Dex1 finger joint limits (m) in the URDF
// measured from Dex1_finger_link_1/2.STL: inner jaw faces at |y| = 22.9 mm at joint 0, so the fingertip gap is
// 2 * (22.9 mm + q): 45.8 mm at q = 0, 5.8 mm at the URDF closed limit (-0.02), 94.8 mm at the open limit.
// The real gripper closes fully (0 mm) and opens to 83 mm, so the visuals go past the URDF closed limit.
const FINGER_INNER = 0.0229, DEX1_OPEN_MM = 83;
const gapToQ = mm => mm / 2000 - FINGER_INNER;        // fingertip gap (mm) -> finger joint value (m)
const GRIP_OPEN_VALUE = 5.4;                         // recorded gripper state: 0 closed .. 5.4 open
const D435_IN_TORSO = new THREE.Vector3(0.0576235, 0.01753, 0.42987);
// XR (y up, -z forward) <- robot (x forward, y left, z up); Matrix4.set is row-major
const T_XR_ROBOT = new THREE.Matrix4().set(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1);
const T_EE_DEX1_BASE = new THREE.Matrix4().makeTranslation(0.0415 - 0.05, 0, 0);
const GHOST_LOCAL = new THREE.Matrix4().multiplyMatrices(T_XR_ROBOT, T_EE_DEX1_BASE); // controller -> ghost base
const GHOST_LOCAL_INV = GHOST_LOCAL.clone().invert();
const COLOR_IDLE = new THREE.Color(0x4fc3f7), COLOR_FOLLOW = new THREE.Color(0x66ff88);
const COLOR_ROBOT = new THREE.Color(0xcfd8e3);
// gripper force / width sliders at the thumb-rest anchor (2026-11-10). Ranges depend on the end effector;
// Dex1-1: force from the Unitree spec (unitree.com/Dex1-1), 5-120 N adjustable (not measured by us).
// Width: 83 mm measured at the fingertips on the real gripper (2026-11-10); spec says 90 mm stroke, URDF 2 x 44.5 mm.
// ?gripper=rubber_hand: ghost and robot show the rubber hand, no sliders.
const GRIPPER_PROFILES = {
  dex1: { name: 'Dex1-1', force: { min: 5, max: 120, unit: 'N' }, width: { min: 0, max: 83, unit: 'mm' } },
  rubber_hand: { name: 'Rubber hand', force: null, width: null },   // passive hand (dance / motion): nothing to set
};
const urlGripper = new URLSearchParams(location.search).get('gripper');   // ?gripper= wins over the server's --gripper
let gripperKind = GRIPPER_PROFILES[urlGripper] ? urlGripper : 'dex1';
const gripSet = { left: { force: 0.5, width: 1.0 }, right: { force: 0.5, width: 1.0 } };   // 0..1
const SCALE_MIN = 0.5, SCALE_MAX = 2.0;              // scene scale limits (smaller made the operator lose orientation)

// ---------- renderer / scene
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const VR_BG = new THREE.Color(0x14171c);
scene.background = VR_BG;
scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(1, 3, 2);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.02, 50);
camera.position.set(0.0, 1.75, 1.3);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 1.0, -0.5);
orbit.update();

// workspace = operator anchor; robotFrame puts the robot's head camera at the workspace origin
const workspace = new THREE.Group();
workspace.position.set(0, 1.35, 0);                 // desktop preview; re-anchored to the head when XR starts
scene.add(workspace);
const robotFrame = new THREE.Group();
robotFrame.quaternion.setFromRotationMatrix(T_XR_ROBOT);
workspace.add(robotFrame);

// ---------- panels
function textCanvas(w, h) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
}
function placeholderTexture(label) {
  const c = textCanvas(640, 480), g = c.getContext('2d');
  g.fillStyle = '#20252d'; g.fillRect(0, 0, 640, 480);
  g.fillStyle = '#9aa0a6'; g.font = '36px system-ui'; g.textAlign = 'center'; g.fillText(label, 320, 250);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function makePanel(w, h, label) {
  const mat = new THREE.MeshBasicMaterial({ map: placeholderTexture(label), side: THREE.DoubleSide, toneMapped: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color: 0x4fc3f7 }));
  mesh.add(frame);
  mesh.userData.frame = frame;
  return mesh;
}
const headPanel = makePanel(0.8, 0.6, 'Head camera: waiting');
headPanel.position.set(0, 0.42, -1.0);
headPanel.rotation.x = Math.atan2(0.42, 1.0);
workspace.add(headPanel);

const hudCanvas = textCanvas(1024, 160);
const hudTex = new THREE.CanvasTexture(hudCanvas); hudTex.colorSpace = THREE.SRGBColorSpace;
const hud = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.125),
  new THREE.MeshBasicMaterial({ map: hudTex, transparent: true, toneMapped: false, depthTest: false }));
hud.renderOrder = 10;
hud.position.set(0, 0.80, -1.0);
hud.rotation.x = Math.atan2(0.80, 1.0);
workspace.add(hud);

const wristPanels = {
  left: makePanel(0.32, 0.24, 'Left wrist camera'),
  right: makePanel(0.32, 0.24, 'Right wrist camera'),
};
for (const s of SIDES) { wristPanels[s].visible = false; scene.add(wristPanels[s]); }

// images: 0 head, 1 left wrist, 2 right wrist
const imageTargets = { 0: headPanel, 1: wristPanels.left, 2: wristPanels.right };
const decoding = {};
async function showJpeg(tag, bytes) {
  const panel = imageTargets[tag];
  if (!panel || decoding[tag]) return;          // drop frames while the previous one is still decoding
  decoding[tag] = true;
  try {
    const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), { imageOrientation: 'flipY' });
    let tex = panel.material.map;
    if (!(tex.userData.live)) {
      tex.dispose();
      tex = new THREE.Texture(); tex.colorSpace = THREE.SRGBColorSpace; tex.flipY = false; tex.userData.live = true;
      if (tag === 2) { tex.center.set(0.5, 0.5); tex.rotation = Math.PI; }   // right D405 is mounted rolled 180 deg:
      // shown upright here only, the recorded / streamed image stays as the camera gives it (like the real robot)
      panel.material.map = tex; panel.material.needsUpdate = true;
    } else if (tex.image && tex.image.close) tex.image.close();
    tex.image = bmp; tex.needsUpdate = true;
  } catch (e) { console.warn('jpeg decode', e); }
  decoding[tag] = false;
}

// ---------- point cloud (pelvis frame: lands in front of the digital robot and scales / moves with the scene)
// Server tag 4 (v1.2): [4][uint32 n][int16 xyz mm][uint8 rgb]; tag 3 (float32 xyz) still understood.
// Left stick press (L3, alone) shows / hides it; it fades when no new cloud came for CLOUD_STALE_S.
const CLOUD_POINT_SIZE = 0.0045;                        // m, with size attenuation (7 mm voxels since 2026-11-21)
const CLOUD_STALE_S = 2.0;
const cloudMat = new THREE.PointsMaterial({ size: CLOUD_POINT_SIZE, vertexColors: true, sizeAttenuation: true,
  transparent: true, opacity: 1.0 });
const cloud = new THREE.Points(new THREE.BufferGeometry(), cloudMat);
cloud.frustumCulled = false;
robotFrame.add(cloud);
let cloudSeen = false, cloudOn = true, cloudLast = 0, cloudCount = 0;
function showCloud(buf, tag) {
  const n = new DataView(buf).getUint32(1, true);
  let xyz;
  if (tag === 4) {
    const mm = new Int16Array(buf.slice(5, 5 + 6 * n));
    xyz = new Float32Array(3 * n);
    for (let i = 0; i < 3 * n; i++) xyz[i] = mm[i] * 0.001;
  } else xyz = new Float32Array(buf.slice(5, 5 + 12 * n));
  const off = 5 + (tag === 4 ? 6 : 12) * n;
  const rgb = new Uint8Array(buf.slice(off, off + 3 * n));
  const g = cloud.geometry;
  g.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  g.setAttribute('color', new THREE.BufferAttribute(rgb, 3, true));
  g.computeBoundingSphere();
  cloudSeen = true; cloudLast = performance.now(); cloudCount = n;
}
function updateCloudVisibility() {
  cloud.visible = cloudOn && cloudSeen;
  const age = (performance.now() - cloudLast) / 1000;
  cloudMat.opacity = age < CLOUD_STALE_S ? 1.0 : 0.25;                // old cloud: faded, still shows where things were
}

// ---------- floor ring (robot frame: z up, so the XZ grid helper is turned into the XY plane)
const floorRing = new THREE.PolarGridHelper(0.6, 8, 3, 48, 0x3c4654, 0x3c4654);   // robot's own floor (under the feet)
floorRing.rotation.x = Math.PI / 2;
floorRing.position.z = -0.79;
robotFrame.add(floorRing);

// ---------- real floor (XR local-floor: world y = 0 is the real floor). Shadow + contact ring under the robot's feet:
// ring green = feet on the floor (within FEET_ON_FLOOR_M), orange = floating / sunk; HUD shows the height.
const FEET_ON_FLOOR_M = 0.01, FEET_SNAP_M = 0.03;
const floorShadow = new THREE.Mesh(new THREE.CircleGeometry(0.28, 48),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28, depthWrite: false }));
floorShadow.rotation.x = -Math.PI / 2; floorShadow.renderOrder = 1; scene.add(floorShadow);
const contactRing = new THREE.Mesh(new THREE.RingGeometry(0.29, 0.31, 64),
  new THREE.MeshBasicMaterial({ color: 0xffa726, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
contactRing.rotation.x = -Math.PI / 2; contactRing.renderOrder = 2; scene.add(contactRing);
const floorGrid = new THREE.GridHelper(2.0, 20, 0x56606e, 0x3c4654);
floorGrid.material.transparent = true; floorGrid.material.opacity = 0.35; scene.add(floorGrid);
const feetW = new THREE.Vector3();
function feetWorld() { floorRing.updateWorldMatrix(true, false); return floorRing.getWorldPosition(feetW); }
let feetHeight = null;                                  // m above the real floor (world y = 0), null until placed
function updateFloorMarks() {
  if (!robotPlaced) { floorShadow.visible = contactRing.visible = floorGrid.visible = false; return; }
  const f = feetWorld(), sc = workspace.scale.x;
  feetHeight = f.y;
  floorShadow.visible = contactRing.visible = floorGrid.visible = true;
  floorShadow.position.set(f.x, 0.002, f.z); floorShadow.scale.setScalar(sc);
  contactRing.position.set(f.x, 0.003, f.z); contactRing.scale.setScalar(sc);
  floorGrid.position.set(f.x, 0.001, f.z); floorGrid.scale.setScalar(sc);
  contactRing.material.color.set(Math.abs(feetHeight) <= FEET_ON_FLOOR_M ? 0x66ff88 : 0xffa726);
}

// ---------- robot model
let robot = null, robotPlaced = false;
// robot 60 % opaque: what is behind it stays visible (user's choice 2026-11-10)
const robotMat = new THREE.MeshStandardMaterial({ color: COLOR_ROBOT, metalness: 0.1, roughness: 0.7,
  transparent: true, opacity: 0.6 });
const stl = new STLLoader();
const urdfLoader = new URDFLoader();
urdfLoader.loadMeshCb = (path, manager, done) => {
  // wrapped in a Group: URDFLoader replaces the material of a bare Mesh with the URDF's colour
  stl.load(path, geo => done(new THREE.Group().add(new THREE.Mesh(geo, robotMat))), undefined, err => done(null, err));
};
const robotRubber = {};
function syncGripperKind() {      // show Dex1 or rubber hand on the robot and the ghosts
  const rubberOn = gripperKind === 'rubber_hand';
  for (const s of SIDES) {
    const g = ghosts[s];
    g.base.visible = g.f1.visible = g.f2.visible = !rubberOn; g.rubber.visible = rubberOn;
    if (robot) {
      const dex = robot.links[`${s}_dex1_base_link`]; if (dex) dex.visible = !rubberOn;   // fingers are its children
      if (robotRubber[s]) robotRubber[s].visible = rubberOn;
    }
  }
}
urdfLoader.load(URDF_URL, r => {
  robot = r;
  r.traverse(o => { if (o.isURDFCollider) o.visible = false; });
  for (const s of SIDES) {
    const wrist = r.links[`${s}_wrist_yaw_link`];
    if (!wrist) continue;
    const grp = new THREE.Group(); grp.position.set(0.0415, s === 'left' ? 0.003 : -0.003, 0); grp.visible = false;
    stl.load(`/robot/meshes/${s}_rubber_hand.STL`, geo => grp.add(new THREE.Mesh(geo, robotMat)));
    wrist.add(grp); robotRubber[s] = grp;
    r.links[`${s}_dex1_base_link`]?.add(makeWristCam(s, robotMat));   // hidden with the Dex1 in rubber-hand mode
  }
  robotFrame.add(r);
  setStatus('Robot model loaded', 'ok');
}, undefined, e => setStatus('Robot model failed to load: ' + e, 'bad'));

function applyState(st) {
  if (!robot) return;
  ARM_JOINTS.forEach((j, i) => robot.setJointValue(j, st.arm?.[i] ?? 0));
  LEG_JOINTS.forEach((j, i) => robot.setJointValue(j, st.legs?.[i] ?? 0));
  WAIST_JOINTS.forEach((j, i) => robot.setJointValue(j, st.waist?.[i] ?? 0));
  SIDES.forEach((s, i) => {
    const q = gapToQ(DEX1_OPEN_MM * THREE.MathUtils.clamp((st.grip?.[i] ?? 0) / GRIP_OPEN_VALUE, 0, 1));   // 0 = touching
    for (const k of [1, 2]) { const jt = robot.joints[`${s}_dex1_finger_joint_${k}`]; if (jt) jt.ignoreLimits = true; }
    robot.setJointValue(`${s}_dex1_finger_joint_1`, q);
    robot.setJointValue(`${s}_dex1_finger_joint_2`, q);
  });
  if (!robotPlaced) placeRobot();
}
function placeRobot() {           // head camera of the robot -> workspace origin (operator's eyes)
  const torso = robot.links['torso_link'];
  if (!torso) return;
  robot.updateMatrixWorld(true);
  const rel = robot.matrixWorld.clone().invert().multiply(torso.matrixWorld);
  const cam = D435_IN_TORSO.clone().applyMatrix4(rel).applyQuaternion(robotFrame.quaternion);
  robotFrame.position.copy(cam.negate());
  // floor ring under the robot's feet: a fixed reference when the scene is moved or scaled
  const foot = robot.links['left_ankle_roll_link'];
  if (foot) {
    const f = new THREE.Vector3().setFromMatrixPosition(robot.matrixWorld.clone().invert().multiply(foot.matrixWorld));
    floorRing.position.set(0, 0, f.z - 0.035);
  }
  robotPlaced = true;
}

// ---------- Dex1 wrist camera model (ghost + digital robot) = the REAL robot's wrist camera (user photos 2026-11-21
// with ruler + protractor, and the real wrist frames of the Unitree G1_Stack_Block dataset): a silver stereo box
// (~42 x 42 x 23 mm, two lenses ~16 mm apart, USB-C on the side; most likely an Intel RealSense D405, colour from the
// left imager) on a white 3D-printed bracket that clamps around the Dex1 motor body. Photo-based estimate of the lens
// midpoint: see WRIST_CAM_POS below. The sim wrist cameras (camera_configs.py) use exactly this pose:
// hand_base (0, 0.035, 0.078), quat wxyz (-0.47716, 0.87882, 0, 0). Bracket shape is approximate (from the photos).
const SHOW_WRIST_CAM_FRUSTUM = true;                    // false = no view lines from the wrist camera
// 2026-11-21 (v3, FIT to the real D405 images of the g1_camera_twin capture: floor plane + finger silhouette,
// residuals ~0.7 deg / ~5 mm / finger IoU ~0.75, est. +-5 mm, +-2 deg). Same poses as camera_configs.py, converted from
// hand_base (ROS optical) to the Dex1 base frame. The RIGHT camera is rolled 180 deg (USB-C cable exits outward).
const WRIST_CAM_POS = { left: new THREE.Vector3(0.059, 0.0137, 0.0627), right: new THREE.Vector3(0.0512, -0.0049, 0.0611) };
const WRIST_CAM_QUAT = {                                // camera (x image right, y image down, z view) in the Dex1 base frame
  left: new THREE.Quaternion(0.61279, -0.61516, 0.37009, -0.3303),
  right: new THREE.Quaternion(-0.61404, -0.60788, -0.34434, -0.36725) };
// FOV = the sim camera (aperture 20 x 15 mm, f 12 mm = 80 x 64 deg); D405 colour is ~87 x 58 deg (not used yet)
const WRIST_CAM_HFOV = THREE.MathUtils.degToRad(72.5), WRIST_CAM_VFOV = THREE.MathUtils.degToRad(57.6);   // D405 640x480
const WRIST_CAM_BOX = { w: 0.042, h: 0.042, d: 0.023, lensGap: 0.016 };
const camMats = {                                       // digital robot: real colours, see-through like the robot
  silver: new THREE.MeshStandardMaterial({ color: 0xc3c8ce, metalness: 0.6, roughness: 0.35, transparent: true, opacity: 0.6 }),
  white: new THREE.MeshStandardMaterial({ color: 0xf2f0ea, metalness: 0.0, roughness: 0.8, transparent: true, opacity: 0.6 }),
  lens: new THREE.MeshStandardMaterial({ color: 0x15181c, metalness: 0.2, roughness: 0.3, transparent: true, opacity: 0.8 }),
};
function boxBetween(a, b, w, t, mat, up) {              // thin bar from a to b (w across, t thick, t along 'up')
  const len = a.distanceTo(b), mesh = new THREE.Mesh(new THREE.BoxGeometry(w, t, len), mat);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().lookAt(a, b, up));
  return mesh;
}
function makeWristCam(s, mat, frustumColor) {
  const ghost = frustumColor !== undefined;
  const mSilver = ghost ? mat : camMats.silver, mWhite = ghost ? mat : camMats.white, mLens = ghost ? mat : camMats.lens;
  const B = WRIST_CAM_BOX, holder = new THREE.Group(), y0 = WRIST_CAM_POS[s].y;
  const cam = new THREE.Group();                          // origin = lens midpoint on the front face
  cam.position.copy(WRIST_CAM_POS[s]);
  cam.quaternion.copy(WRIST_CAM_QUAT[s]);
  holder.add(cam);
  const box = new THREE.Mesh(new THREE.BoxGeometry(B.w, B.h, B.d), mSilver); box.position.z = -B.d / 2; cam.add(box);
  const face = new THREE.Mesh(new THREE.BoxGeometry(B.w - 0.006, B.h - 0.006, 0.0015), mLens);   // black front glass
  face.position.z = 0.0002; cam.add(face);
  for (const sx of [-1, 1]) {                             // two lens rings + dark lenses on the front
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.0045, 0.0012, 8, 24), mSilver);
    ring.position.set(sx * B.lensGap / 2, 0, 0.0012); cam.add(ring);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.0034, 0.0034, 0.002, 20), mLens);
    lens.rotation.x = Math.PI / 2; lens.position.set(sx * B.lensGap / 2, 0, 0.0008); cam.add(lens);
  }
  const plug = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.008, 0.016), ghost ? mat : camMats.lens);   // USB-C, outer side
  cam.updateMatrix();
  const outer = new THREE.Vector3(0, s === 'left' ? 1 : -1, 0).applyQuaternion(cam.quaternion.clone().invert());  // Dex1 +-y in camera
  plug.position.copy(outer.multiplyScalar(B.w / 2 + 0.006)).add(new THREE.Vector3(0, 0, -B.d / 2)); cam.add(plug);
  const cradle = new THREE.Mesh(new THREE.BoxGeometry(B.w + 0.006, B.h + 0.006, 0.005), mWhite);   // bracket plate behind
  cradle.position.z = -B.d - 0.0025; cam.add(cradle);
  // white bracket: clamp ring around the motor body at the wrist end, an arm rising behind it, bending forward to the
  // bottom of the camera's back plate (shape approximated from the photos)
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.037, 0.005, 10, 40), mWhite);
  ring.rotation.y = Math.PI / 2; ring.position.set(0.006, y0, 0); holder.add(ring);
  cam.updateMatrix();
  const plateBottom = new THREE.Vector3(0, B.h / 2 - 0.004, -B.d - 0.0025).applyMatrix4(cam.matrix);  // +y = image down
  const p0 = new THREE.Vector3(0.004, y0, 0.040), p1 = new THREE.Vector3(-0.002, y0, 0.056);
  const up = new THREE.Vector3(0, 1, 0);
  holder.add(boxBetween(p0, p1, 0.007, 0.03, mWhite, up), boxBetween(p1, plateBottom, 0.007, 0.03, mWhite, up));
  if (SHOW_WRIST_CAM_FRUSTUM && ghost) {
    const L = 0.10, hx = L * Math.tan(WRIST_CAM_HFOV / 2), hy = L * Math.tan(WRIST_CAM_VFOV / 2);
    const c = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([x, y]) => new THREE.Vector3(x, y, L));
    const pts = [];
    for (let i = 0; i < 4; i++) pts.push(new THREE.Vector3(0, 0, 0.002), c[i], c[i], c[(i + 1) % 4]);
    cam.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: frustumColor, transparent: true, opacity: 0.6 })));
  }
  return holder;
}

// ---------- ghost grippers (60 % opaque, tinted: blue = idle, green = arm follows)
function ghostMaterial() {
  return new THREE.MeshStandardMaterial({ color: COLOR_IDLE.clone(), metalness: 0.1, roughness: 0.6,
    transparent: true, opacity: 0.6, depthWrite: false });
}
const ghosts = {};
for (const s of SIDES) {
  const mat = ghostMaterial();
  const root = new THREE.Group(); root.matrixAutoUpdate = false; root.visible = false;
  const base = new THREE.Group(), f1 = new THREE.Group(), f2 = new THREE.Group();
  root.add(base, f1, f2);
  scene.add(root);
  // thumb-rest anchor: the fixed point (yellow) and a line to the controller while the ghost is being moved
  const anchor = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffd54f, depthTest: false, transparent: true }));
  anchor.renderOrder = 11; anchor.visible = false; scene.add(anchor);
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineBasicMaterial({ color: 0xffd54f, depthTest: false, transparent: true }));
  line.renderOrder = 11; line.frustumCulled = false; line.visible = false; scene.add(line);
  // rubber hand: URDF hand_palm_joint is wrist_yaw + (0.0415, +-0.003, 0); Dex1 base is wrist_yaw + (0.0415, 0, 0)
  const rubber = new THREE.Group(); rubber.position.set(0, s === 'left' ? 0.003 : -0.003, 0); root.add(rubber);
  stl.load(`/robot/meshes/${s}_rubber_hand.STL`, geo => rubber.add(new THREE.Mesh(geo, mat)));
  ghosts[s] = { root, base, f1, f2, rubber, mat, anchor, line, offset: new THREE.Matrix4(), frozen: false, sliderUsed: false };
  for (const [grp, file] of [[base, 'Dex1_base_link'], [f1, 'Dex1_finger_link_1'], [f2, 'Dex1_finger_link_2']]) {
    stl.load(`/robot/meshes/${file}.STL`, geo => grp.add(new THREE.Mesh(geo, mat)));
  }
  base.add(makeWristCam(s, mat, 0xffd54f));             // shown / hidden together with the Dex1 base
}
// ---------- force / width slider panel (shown above the yellow dot while the thumb rest is touched)
// slider panel geometry (m, panel-local); the canvas has the plane's aspect, bars are drawn from these same numbers
const SL = { w: 0.18, h: 0.11, x0: -0.075, x1: 0.075, yForce: 0.02, yWidth: -0.03 };   // bars 5 cm apart
const SL_PX = 512 / SL.w;                                   // canvas pixels per metre
const SL_W = 512, SL_H = Math.round(SL.h * SL_PX);
function makeSliderPanel() {
  const canvas = textCanvas(SL_W, SL_H);
  const tex = new THREE.CanvasTexture(canvas); tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(SL.w, SL.h),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, depthTest: false, side: THREE.DoubleSide }));
  mesh.renderOrder = 12; mesh.visible = false; scene.add(mesh);
  return { mesh, canvas, tex, key: '', active: null, until: 0 };
}
const sliders = { left: makeSliderPanel(), right: makeSliderPanel() };
function sliderValue(range, v) { return range.min + (range.max - range.min) * v; }
function drawSliders(s) {
  const P = GRIPPER_PROFILES[gripperKind], sp = sliders[s], v = gripSet[s];
  const key = `${gripperKind}${v.force.toFixed(3)}${v.width.toFixed(3)}${sp.active}`;
  if (key === sp.key) return;
  sp.key = key;
  const g = sp.canvas.getContext('2d'), W = SL_W, H = SL_H;
  const px = x => (x + SL.w / 2) * SL_PX, py = y => (SL.h / 2 - y) * SL_PX;   // panel metres -> canvas pixels
  g.clearRect(0, 0, W, H);
  g.fillStyle = 'rgba(15,17,21,0.85)'; g.fillRect(0, 0, W, H);
  g.font = '26px system-ui'; g.fillStyle = '#e8eaed';
  if (!P.force) { g.fillText(`${P.name}: no force / width to set`, 24, H / 2 + 9); sp.tex.needsUpdate = true; return; }
  const row = (label, range, val, y, active) => {
    const yy = py(y);
    g.fillStyle = '#e8eaed';
    g.fillText(`${label}  ${Math.round(val * 100)} %   ${sliderValue(range, val).toFixed(1)} ${range.unit}`, 24, yy - 22);
    const a = px(SL.x0), b = px(SL.x1);
    g.fillStyle = '#3c4654'; g.fillRect(a, yy - 4, b - a, 8);
    g.fillStyle = active ? '#66ff88' : '#ffd54f'; g.fillRect(a, yy - 4, (b - a) * val, 8);
    g.beginPath(); g.arc(a + (b - a) * val, yy, active ? 16 : 12, 0, 2 * Math.PI); g.fill();
    g.fillStyle = '#9aa0a6'; g.font = '18px system-ui';
    g.fillText(`${range.min} ${range.unit}`, a, yy + 30);
    const t = `${range.max} ${range.unit}`; g.fillText(t, b - g.measureText(t).width, yy + 30);
    g.font = '26px system-ui';
  };
  row('Force', P.force, v.force, SL.yForce, sp.active === 'force');
  row('Width', P.width, v.width, SL.yWidth, sp.active === 'width');
  sp.tex.needsUpdate = true;
}
// same hand: while the thumb rest is touched, the bar nearest to the controller (within 6 cm) lights up;
// hold grip there = that bar is grabbed; move the controller right (operator's view) = more, left = less;
// let go of grip = value stays. Grip with no bar near = nothing changes.
const SLIDER_FULL_M = 0.20;                               // 20 cm hand movement = 0 .. 100 %
const SLIDER_NEAR_M = 0.06;
const headRight = new THREE.Vector3(), cLocal = new THREE.Vector3();
function nearestBar(sp, cPos) {
  sp.mesh.updateMatrixWorld(true);
  cLocal.copy(cPos); sp.mesh.worldToLocal(cLocal);
  const dxOut = Math.max(0, SL.x0 - cLocal.x, cLocal.x - SL.x1);
  let best = null, bestD = SLIDER_NEAR_M;
  for (const [which, y] of [['force', SL.yForce], ['width', SL.yWidth]]) {
    const d = Math.hypot(dxOut, cLocal.y - y, cLocal.z);
    if (d < bestD) { bestD = d; best = which; }
  }
  return best;
}
function sendGripSettings(s) {
  const P = GRIPPER_PROFILES[gripperKind];
  if (!(ws && ws.readyState === 1)) return;
  ws.send(JSON.stringify({ t: 'grip_settings', side: s, gripper: gripperKind, force: gripSet[s].force, width: gripSet[s].width,
    force_n: P.force ? sliderValue(P.force, gripSet[s].force) : null, width_mm: P.width ? sliderValue(P.width, gripSet[s].width) : null }));
}
function updateSliders(s, rest, squeeze, cPos) {
  const sp = sliders[s], g = ghosts[s], P = GRIPPER_PROFILES[gripperKind];
  // visible only while the thumb rest is touched, or while a grabbed slider is still held with grip (2026-11-21)
  const holding = !!sp.grab && squeeze;
  sp.mesh.visible = rest || holding;
  if (!sp.mesh.visible) { if (sp.grab) sendGripSettings(s); sp.active = null; sp.grab = null; return; }
  // panel right above the yellow dot: the width bar is ~4 cm and the force bar ~9 cm above the hand
  if (rest) sp.mesh.position.copy(g.anchor.position).add(new THREE.Vector3(0, 0.07, 0));
  const near = rest && P.force ? nearestBar(sp, cPos) : null;
  if (squeeze && P.force && (rest || sp.grab)) {
    if (sp.grab === undefined || sp.grab === null) {
      sp.grab = near ? { which: near, start: cPos.clone(), v0: gripSet[s][near] } : false;
      if (sp.grab) g.sliderUsed = true;                   // this thumb-rest touch adjusts a slider, it does not move the ghost
    }
  } else {
    if (sp.grab) sendGripSettings(s);
    sp.grab = null;                                       // false = grip pressed away from the bars: ignore until released
  }
  if (sp.grab) {
    headRight.set(1, 0, 0).applyQuaternion(headQuat); headRight.y = 0; headRight.normalize();
    const dx = tmpV.copy(cPos).sub(sp.grab.start).dot(headRight);
    gripSet[s][sp.grab.which] = THREE.MathUtils.clamp(sp.grab.v0 + dx / SLIDER_FULL_M, 0, 1);
    sp.active = sp.grab.which;
  } else sp.active = near;
  drawSliders(s);
}

function setGhostFingers(s, gapMm) {                // visual fingertip gap in mm (0 = touching)
  const q = gapToQ(THREE.MathUtils.clamp(gapMm, 0, 94.8));
  ghosts[s].f1.position.set(0, -q, 0);     // finger joint 1 axis (0,-1,0)
  ghosts[s].f2.position.set(0, q, 0);      // finger joint 2 axis (0, 1,0)
}

// ---------- WebSocket
let ws = null, lastState = null, sendCmd = () => {};
function connect() {
  ws = new WebSocket(`wss://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { setStatus('Connected to the robot server', 'ok'); enableButtons(); };
  ws.onclose = () => { setStatus('Server connection lost, retrying…', 'bad'); setTimeout(connect, 1500); };
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') {
      const m = JSON.parse(ev.data);
      if (m.t === 'state') { lastState = m; if (!GRIPPER_PROFILES[urlGripper] && GRIPPER_PROFILES[m.gripper]) gripperKind = m.gripper; applyState(m); }
      return;
    }
    const buf = ev.data, tag = new Uint8Array(buf, 0, 1)[0];
    if (tag === 3 || tag === 4) showCloud(buf, tag); else showJpeg(tag, new Uint8Array(buf, 1));
  };
  sendCmd = cmd => { if (ws.readyState === 1) ws.send(JSON.stringify({ t: 'cmd', cmd })); toast(CMD_LABEL[cmd] || cmd); };
}
const CMD_LABEL = { toggle_episode: 'Start / end episode', back: 'Back (task page comes later)',
  next_episode: 'Next episode', prev_episode: 'Previous episode' };

// ---------- HUD
let toastText = '', toastUntil = 0, hudKey = '';
function toast(t) { toastText = t; toastUntil = performance.now() + 2000; }
function fmt(s) { s = Math.max(0, Math.floor(s)); return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; }
function drawHud() {
  const st = lastState;
  const l1 = st ? `${String(st.source ?? 'server').toUpperCase()} · ${st.episode ?? '-'}  (${(st.episode_idx ?? 0) + 1}/${st.episode_count ?? '?'})  ${APP_BUILD}` : 'No data yet';
  const l2 = !st ? '' : st.running ? `● ${fmt(st.elapsed ?? 0)}` : '■ stopped   right stick → to start';
  const cl = !cloudSeen ? '' : !cloudOn ? '   cloud off (L3)'
    : (performance.now() - cloudLast) / 1000 > CLOUD_STALE_S ? '   cloud: no update' : `   cloud ${Math.round(cloudCount / 1000)}k pts`;
  // live sim (server --source sim): per-arm clutch state; the server's notice (e.g. clutch refused) wins over toasts
  const fc = c => !c ? '-' : c.clutch === 'follow' ? 'FOLLOW' : c.clutch === 'rejected' ? `too far ${c.dist_cm ?? '?'} cm` : 'hold';
  const hv = st?.haptic && hapticsOn ? `  vib ${(st.haptic.left || 0).toFixed(1)}/${(st.haptic.right || 0).toFixed(1)}` : '';
  const lv = st?.live ? `   LIVE  L ${fc(st.live.left)} · R ${fc(st.live.right)}${hapticsOn ? (hapticsMissing ? '  (no vibration device)' : '') : '  (haptics off, R3)'}${hv}` : '';
  const noDepth = st?.source === 'sim' && st?.cloud_expected && !cloudSeen ? 'no depth: start the sim with G1_REAL_HEAD_DEPTH=1' : '';
  const l3 = st?.notice ? st.notice : performance.now() < toastUntil ? toastText : noDepth;
  const fh = feetHeight === null ? '' : Math.abs(feetHeight) <= FEET_ON_FLOOR_M ? '   feet on floor'
    : `   feet ${feetHeight > 0 ? '+' : ''}${Math.round(feetHeight * 100)} cm ${feetHeight > 0 ? 'above' : 'below'} floor`;
  const key = l1 + l2 + cl + lv + fh + l3 + (st && st.running);
  if (key === hudKey) return;
  hudKey = key;
  const g = hudCanvas.getContext('2d');
  g.clearRect(0, 0, 1024, 160);
  g.fillStyle = 'rgba(15,17,21,0.8)'; g.fillRect(0, 0, 1024, 160);
  g.font = '34px system-ui'; g.fillStyle = '#e8eaed'; g.fillText(l1, 24, 48);
  g.fillStyle = st && st.running ? '#ef5350' : '#9aa0a6'; g.fillText(l2, 24, 96);
  g.fillStyle = '#9aa0a6'; g.fillText(cl, 24 + g.measureText(l2).width, 96);
  if (lv) {
    const rej = /too far/.test(lv), fol = /FOLLOW/.test(lv);
    g.fillStyle = rej ? '#ffa726' : fol ? '#66ff88' : '#9aa0a6';
    g.fillText(lv, 24 + g.measureText(l2 + cl).width, 96);
  }
  g.fillStyle = feetHeight !== null && Math.abs(feetHeight) <= FEET_ON_FLOOR_M ? '#66ff88' : '#ffa726';
  g.fillText(fh, 24 + g.measureText(l1).width, 48);
  g.fillStyle = '#4fc3f7'; g.fillText(l3, 24, 144);
  hudTex.needsUpdate = true;
}

// ---------- XR session
const btnVR = document.getElementById('btn-vr'), btnAR = document.getElementById('btn-ar');
function setStatus(t, cls) { const el = document.getElementById('status'); el.textContent = t; el.className = cls || ''; }
async function enableButtons() {
  if (!navigator.xr) { setStatus('Connected. This browser has no WebXR: desktop preview only (mouse to orbit).', 'ok'); return; }
  btnVR.disabled = !(await navigator.xr.isSessionSupported('immersive-vr'));
  btnAR.disabled = !(await navigator.xr.isSessionSupported('immersive-ar'));
}
let anchored = false;
async function startXR(mode) {
  const session = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor'] });
  scene.background = mode === 'immersive-ar' ? null : VR_BG;
  anchored = false;
  await renderer.xr.setSession(session);
  session.addEventListener('end', resetXRState);
  session.addEventListener('inputsourceschange', e =>
    pageLog(`inputsourceschange: +[${e.added.map(srcInfo).join('; ')}] -[${e.removed.map(srcInfo).join('; ')}]`));
  pageLog(`XR session started (${mode}): ${[...session.inputSources].map(srcInfo).join('; ') || 'no inputs yet'}`);
}
function resetXRState() {          // XR session ended: nothing half-done carries over into the next session
  scene.background = VR_BG;
  for (const s of SIDES) {
    const g = ghosts[s], sp = sliders[s];
    g.root.visible = g.anchor.visible = g.line.visible = false;
    g.frozen = false; g.sliderUsed = false; g.placed = false;
    sp.mesh.visible = false; sp.until = 0; sp.grab = null; sp.active = null;
    wristPanels[s].visible = false;
    prev[s] = {}; stickArmed[s] = true;
  }
  twoHand = null; lastBothPress = 0;
  for (const s of SIDES) { handSrc[s] = null; handObj[s].userData.src = null; delete controllers[s]; }
  workspace.scale.setScalar(1);
}
btnVR.onclick = () => startXR('immersive-vr');
btnAR.onclick = () => startXR('immersive-ar');

// 2026-11-22e: the controller state is re-read EVERY frame from session.inputSources by handedness (own pose object per
// side). Before, the page kept the XRInputSource that three.js handed over in its 'connected' event; when the Quest swapped
// the right controller's input source (reconnect / tracking glitch) the old source's gamepad stayed frozen at its last
// values (log 16:07: right "trig 1.00 grip 1 A 1" for ~37 s while the user had let go) -> gripper stayed closed, wrist
// camera stayed visible.
const controllers = {};                                   // side -> handObj[side] while that controller is present
const handObj = { left: new THREE.Object3D(), right: new THREE.Object3D() };
const handSrc = { left: null, right: null };
for (const s of SIDES) { handObj[s].matrixAutoUpdate = false; scene.add(handObj[s]); }
const controllerModels = new XRControllerModelFactory();
controllerModels.path = '/static/vendor/webxr-profiles';   // local copy: works without internet
for (let i = 0; i < 2; i++) {
  const grip = renderer.xr.getControllerGrip(i);          // 3D model of the real controller, follows it exactly
  grip.add(controllerModels.createControllerModel(grip));
  scene.add(grip);
}
function srcInfo(x) { return x ? `${x.handedness}${x.hand ? ' hand' : ''}${x.gamepad ? '' : ' no-gamepad'} [${(x.profiles || []).join(',')}]` : 'none'; }
function resolveInputs(frame) {
  const ref = renderer.xr.getReferenceSpace();
  for (const s of SIDES) {
    let src = null;
    for (const x of frame.session.inputSources) if (x.handedness === s && x.gamepad && !x.hand) { src = x; break; }
    if (src !== handSrc[s]) { pageLog(`${s} controller source ${handSrc[s] ? 'changed' : 'set'} -> ${srcInfo(src)}`); handSrc[s] = src; }
    const o = handObj[s];
    const pose = src ? frame.getPose(src.targetRaySpace, ref) : null;
    if (pose) {
      o.matrix.fromArray(pose.transform.matrix); o.matrix.decompose(o.position, o.quaternion, o.scale);
      o.matrixWorldNeedsUpdate = true; o.userData.src = src; controllers[s] = o;
    } else if (src && o.userData.src === src) controllers[s] = o;   // tracking lost for a moment: last pose, live buttons
    else delete controllers[s];
  }
}

// ---------- page log: errors and input-source changes go to the server log (logs/server_*.log, "[page]")
const pageLogQ = [];
let pageLogLast = '', pageLogRep = 0;
function pageLog(text) {
  console.log('[page]', text);
  if (text === pageLogLast) { if (++pageLogRep > 3) return; } else { pageLogLast = text; pageLogRep = 0; }
  pageLogQ.push(text); if (pageLogQ.length > 50) pageLogQ.shift();
  flushPageLog();
}
function flushPageLog() {
  while (pageLogQ.length && ws && ws.readyState === 1) ws.send(JSON.stringify({ t: 'log', msg: pageLogQ.shift() }));
}
function pageError(where, e) { pageLog(`ERROR in ${where}: ${e?.message || e}${e?.stack ? ' | ' + String(e.stack).split('\n').slice(0, 3).join(' / ') : ''}`); }
window.addEventListener('error', e => pageError('window', e.error || e.message));
window.addEventListener('unhandledrejection', e => pageError('promise', e.reason));

// ---------- controller logic
const prev = { left: {}, right: {} };
const stickArmed = { left: true, right: true };
let twoHand = null, lastBothPress = 0, l3Down = 0, l3Spoiled = false, r3Down = 0, r3Spoiled = false;
const headPos = new THREE.Vector3(), headQuat = new THREE.Quaternion();
const tmpM = new THREE.Matrix4(), tmpV = new THREE.Vector3();

function pad(s) { return controllers[s]?.userData.src?.gamepad; }
function p0(s) { return prev[s]; }
function btnPressed(s, i) { try { return !!pad(s)?.buttons[i]?.pressed; } catch (e) { pageError(`buttons ${s}`, e); return false; } }
// ---------- haptics (live sim): the server sends a level 0..1 per arm (joint limit / contact) and one-shot pulses
// (clutch refused). R3 alone toggles them.
let hapticsOn = true, hapticsMissing = false;
const hapLast = { left: 0, right: 0 }, hapSeq = { left: 0, right: 0 };
function buzz(gp, level, ms) {
  const a = gp?.hapticActuators?.[0];
  try {
    if (a?.pulse) a.pulse(level, ms);
    else if (gp?.vibrationActuator?.playEffect) gp.vibrationActuator.playEffect('dual-rumble', { duration: ms, strongMagnitude: level, weakMagnitude: level });
    else hapticsMissing = true;                         // shown in the HUD
  } catch (e) { hapticsMissing = true; }
}
function doHaptics(s, gp) {
  try { doHapticsInner(s, gp); } catch (e) { pageError(`haptics ${s}`, e); }
}
function doHapticsInner(s, gp) {
  const h = lastState?.haptic;
  if (!hapticsOn || !h) return;
  const now = performance.now();
  const ev = h.pulse?.[s];
  if (ev && ev.seq !== hapSeq[s]) { hapSeq[s] = ev.seq; buzz(gp, ev.level, ev.ms); hapLast[s] = now; return; }
  const lvl = h[s] || 0;
  if (lvl > 0.03 && now - hapLast[s] > 50) { buzz(gp, Math.min(1, lvl), 60); hapLast[s] = now; }
}
function anchorWorkspace(frame) {
  const pose = frame.getViewerPose(renderer.xr.getReferenceSpace());
  if (!pose) return;
  const p = pose.transform.position, o = pose.transform.orientation;
  headQuat.set(o.x, o.y, o.z, o.w);
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(headQuat);
  workspace.position.set(p.x, p.y, p.z);
  workspace.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.atan2(-fwd.x, -fwd.z));
  workspace.scale.setScalar(1);
  anchored = true;
}

function updateControllers(frame) {
  const pose = frame.getViewerPose(renderer.xr.getReferenceSpace());
  if (pose) {
    headPos.set(pose.transform.position.x, pose.transform.position.y, pose.transform.position.z);
    const o = pose.transform.orientation; headQuat.set(o.x, o.y, o.z, o.w);
  }
  const both = SIDES.every(s => btnPressed(s, 3));
  // L3 alone (left stick pressed and released without the right one, < 0.6 s) toggles the point cloud
  const l3 = btnPressed('left', 3), r3 = btnPressed('right', 3);
  if (l3 && !l3Down) { l3Down = performance.now(); l3Spoiled = false; }
  if (l3 && r3) l3Spoiled = true;
  if (r3 && !r3Down) { r3Down = performance.now(); r3Spoiled = false; }
  if (r3 && l3) r3Spoiled = true;
  if (!r3 && r3Down) {
    if (!r3Spoiled && performance.now() - r3Down < 600) { hapticsOn = !hapticsOn; toast(hapticsOn ? 'Haptics on' : 'Haptics off'); }
    r3Down = 0;
  }
  if (!l3 && l3Down) {
    if (!l3Spoiled && performance.now() - l3Down < 600) { cloudOn = !cloudOn; toast(cloudOn ? 'Point cloud on' : 'Point cloud off'); }
    l3Down = 0;
  }
  const msg = { t: 'xr' };
  for (const s of SIDES) {
    try { updateSide(s, both, msg); }
    catch (e) { pageError(`controller ${s}`, e); wristPanels[s].visible = false; }   // one hand's error never blocks the other
  }
  function updateSide(s, both, msg) {
    const c = controllers[s], gp = pad(s), g = ghosts[s];
    if (!c || !gp) { g.root.visible = false; wristPanels[s].visible = false; sliders[s].mesh.visible = false; return; }
    const b = i => gp.buttons[i] || { pressed: false, touched: false, value: 0 };
    const trigger = b(0).value, squeeze = b(1).pressed, clutch = b(4).pressed, resetBtn = b(5).pressed;
    // Thumb on the rest counts only if it lands there BEFORE grip is held (ghost move / slider). Holding the controller by
    // its grip naturally puts the thumb on the rest; that used to freeze the ghost and hide the wrist camera, so the
    // first trigger press with grip held seemed to do nothing (2026-11-22).
    const restRaw = b(6).touched;
    if (restRaw && !p0(s).restRaw) prev[s].restIgnored = squeeze;
    if (!restRaw) prev[s].restIgnored = false;
    const rest = restRaw && !prev[s].restIgnored;
    const [ax, ay] = [gp.axes[2] || 0, gp.axes[3] || 0];
    const p = prev[s];

    // stick commands (not while both sticks are pressed for moving the scene)
    if (!both) {
      if (Math.abs(ax) < 0.3 && Math.abs(ay) < 0.3) stickArmed[s] = true;
      else if (stickArmed[s]) {
        if (s === 'right' && ax > 0.7) { sendCmd('toggle_episode'); stickArmed[s] = false; }
        else if (s === 'right' && ax < -0.7) { sendCmd('back'); stickArmed[s] = false; }
        else if (s === 'left' && ay < -0.7) { sendCmd('next_episode'); stickArmed[s] = false; }
        else if (s === 'left' && ay > 0.7) { sendCmd('prev_episode'); stickArmed[s] = false; }
      }
    }

    // ghost: on the controller with an offset; thumb rest touched = ghost stays put, release = new offset
    c.updateMatrixWorld(true);
    const sc = workspace.scale.x;          // ghost has the same scale as the robot / scene
    const S = new THREE.Matrix4().makeScale(sc, sc, sc);
    if (!g.placed) { g.root.matrix.copy(c.matrixWorld).multiply(g.offset).multiply(GHOST_LOCAL).multiply(S); g.placed = true; }
    // one thumb-rest touch either moves the ghost (release = new offset) or adjusts a slider (offset stays as it was,
    // so reaching for the bars and back never moves the ghost)
    if (rest && !p.rest) { g.frozen = true; g.sliderUsed = false; g.anchor.position.setFromMatrixPosition(c.matrixWorld); }
    if (!rest && p.rest) {
      if (!g.sliderUsed) g.offset.copy(c.matrixWorld).invert().multiply(g.root.matrix)
        .multiply(new THREE.Matrix4().makeScale(1 / sc, 1 / sc, 1 / sc)).multiply(GHOST_LOCAL_INV);
      g.frozen = false; g.sliderUsed = false;
    }
    g.anchor.visible = g.line.visible = rest;
    if (rest) {
      const pts = g.line.geometry.attributes.position;
      pts.setXYZ(0, g.anchor.position.x, g.anchor.position.y, g.anchor.position.z);
      tmpV.setFromMatrixPosition(c.matrixWorld); pts.setXYZ(1, tmpV.x, tmpV.y, tmpV.z); pts.needsUpdate = true;
    }
    if (resetBtn && !p.resetBtn) { g.offset.identity(); toast(`${s} ghost reset`); }
    if (!g.frozen) g.root.matrix.copy(c.matrixWorld).multiply(g.offset).multiply(GHOST_LOCAL).multiply(S);
    g.root.matrixWorldNeedsUpdate = true;
    g.root.visible = true;
    const PW = GRIPPER_PROFILES[gripperKind].width;      // ghost fingertip gap = width setting in mm, trigger closes it
    const openMm = PW ? sliderValue(PW, gripSet[s].width) : DEX1_OPEN_MM;
    setGhostFingers(s, openMm * (1 - trigger));
    updateSliders(s, rest, squeeze, tmpV.setFromMatrixPosition(c.matrixWorld).clone());
    g.mat.color.copy(clutch ? COLOR_FOLLOW : COLOR_IDLE);

    // wrist camera: while the grip button is held (thumb rest not touched), centered right above the ghost
    const wp = wristPanels[s];
    wp.visible = squeeze && !rest && !sliders[s].grab;
    if (wp.visible) {
      tmpV.setFromMatrixPosition(g.root.matrix);
      wp.position.copy(tmpV).add(new THREE.Vector3(0, 0.08 + 0.12, 0));   // panel bottom ~8 cm above the ghost
    }

    // wrist target for the IK (step 2): ghost pose in the robot pelvis frame, Unitree EE convention
    robotFrame.updateMatrixWorld(true);
    tmpM.copy(robotFrame.matrixWorld).invert().multiply(g.root.matrix).multiply(T_EE_DEX1_BASE.clone().invert());
    const P = GRIPPER_PROFILES[gripperKind];
    msg[s] = { ee: tmpM.toArray(), trigger, squeeze, clutch, rest, gripper: gripperKind,
      force: gripSet[s].force, width: gripSet[s].width,                       // 0..1
      force_phys: P.force ? sliderValue(P.force, gripSet[s].force) : null,     // N
      width_phys: P.width ? sliderValue(P.width, gripSet[s].width) : null };   // mm
    Object.assign(p, { rest, resetBtn, restRaw });
    doHaptics(s, gp);
  }

  // both sticks pressed: move / turn / scale the whole scene with the two hands (2026-11-21: feet stay on the floor).
  // Default: the robot's feet are put on the real floor when the gesture starts and stay there (horizontal move, turn,
  // scale about the feet). Holding a grip while pressing both sticks also allows up / down (snaps to the floor < 3 cm).
  if (both && controllers.left && controllers.right) {
    const pl = controllers.left.position.clone(), pr = controllers.right.position.clone();
    const mid = pl.clone().add(pr).multiplyScalar(0.5), v = pr.clone().sub(pl);
    const free = SIDES.some(s => btnPressed(s, 1));
    if (!twoHand) {
      const now = performance.now();
      if (now - lastBothPress < 600) { anchored = false; twoHand = { reset: true }; toast('Scene reset to 100 %'); } // double press
      else {
        if (!free && robotPlaced) { workspace.position.y -= feetWorld().y; workspace.updateMatrixWorld(true); }   // feet onto the floor
        twoHand = { mid0: mid, ang0: Math.atan2(v.x, v.z), dist0: Math.max(v.length(), 0.05), feet0: feetWorld().clone(),
          ws0: new THREE.Matrix4().compose(workspace.position, workspace.quaternion, workspace.scale) };
        toast(free ? 'Moving the scene (grip: up / down too)' : 'Moving the scene, feet on the floor');
      }
      lastBothPress = now;
    } else if (!twoHand.reset) {
      const s0 = twoHand.ws0.getMaxScaleOnAxis();
      let total = THREE.MathUtils.clamp(s0 * v.length() / twoHand.dist0, SCALE_MIN, SCALE_MAX);
      if (Math.abs(total - 1) < 0.07) total = 1;               // snap to 100 %
      const sc = total / s0;
      const dyaw = Math.atan2(v.x, v.z) - twoHand.ang0;
      const d = mid.clone().sub(twoHand.mid0);
      if (!free) d.y = 0;
      const pv = twoHand.feet0;                                // pivot: the feet, so scaling keeps them on the floor
      const m = new THREE.Matrix4().makeTranslation(pv.x + d.x, pv.y + d.y, pv.z + d.z)
        .multiply(new THREE.Matrix4().makeRotationY(dyaw))
        .multiply(new THREE.Matrix4().makeScale(sc, sc, sc))
        .multiply(new THREE.Matrix4().makeTranslation(-pv.x, -pv.y, -pv.z))
        .multiply(twoHand.ws0);
      m.decompose(workspace.position, workspace.quaternion, workspace.scale);
      workspace.updateMatrixWorld(true);
      const fy = feetWorld().y;
      if (!free || Math.abs(fy) < FEET_SNAP_M) workspace.position.y -= fy;   // on the floor / snap to it
      const h = feetWorld().y;
      toast(`Scene ${Math.round(total * 100)} %   ${Math.abs(h) <= FEET_ON_FLOOR_M ? 'feet on floor' : `feet ${Math.round(h * 100)} cm`}`);
    }
  } else twoHand = null;

  if (ws && ws.readyState === 1) { ws.send(JSON.stringify(msg)); flushPageLog(); }
}

// desktop testing keys: space = start/end, arrows up/down = episode, backspace = back
window.addEventListener('keydown', e => {
  if (e.code === 'Space') sendCmd('toggle_episode');
  else if (e.code === 'ArrowUp') sendCmd('next_episode');
  else if (e.code === 'ArrowDown') sendCmd('prev_episode');
  else if (e.code === 'Backspace') sendCmd('back');
});
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// every panel and text faces the operator (upright: Object3D.lookAt keeps world up, no roll)
const eye = new THREE.Vector3();
const bbPos = new THREE.Vector3(), bbTarget = new THREE.Vector3();
function billboardPanels(frame) {
  if (frame) eye.copy(headPos); else eye.copy(camera.position);
  for (const m of [headPanel, hud, wristPanels.left, wristPanels.right, sliders.left.mesh, sliders.right.mesh]) {
    if (!m.visible) continue;
    m.getWorldPosition(bbPos);
    const dx = eye.x - bbPos.x, dz = eye.z - bbPos.z, horiz = Math.hypot(dx, dz);
    if (horiz < 0.02) continue;                         // eye straight above / below: keep the last orientation
    const dy = THREE.MathUtils.clamp(eye.y - bbPos.y, -2 * horiz, 2 * horiz);   // pitch at most ~63 deg
    m.lookAt(bbTarget.set(eye.x, bbPos.y + dy, eye.z));
  }
}
renderer.setAnimationLoop((t, frame) => {
  if (frame) {
    if (!anchored) anchorWorkspace(frame);
    try { resolveInputs(frame); } catch (e) { pageError('resolveInputs', e); }
    updateControllers(frame);
  } else orbit.update();
  syncGripperKind();
  billboardPanels(frame);
  updateCloudVisibility();
  updateFloorMarks();
  drawHud();
  renderer.render(scene, camera);
});
connect();
window.dbg = { cloud, get cloudOn() { return cloudOn; }, set cloudOn(v) { cloudOn = v; }, gripSet, sliders, resetXRState, get gripperKind() { return gripperKind; }, THREE, camera, controllers, updateControllers, resolveInputs, handSrc, wristPanels, orbit, scene, workspace, robotFrame, ghosts, get robot() { return robot; } };
