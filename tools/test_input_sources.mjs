// Deterministic page test (2026-11-22e): the right controller's XRInputSource is swapped while its old gamepad stays
// frozen (trigger 1, grip 1, A 1). The page must follow the NEW source (trigger 0 -> gripper opens, grip 0 -> wrist camera
// hidden), and an exception in one hand must not stop the other.
// Run: node tools/test_input_sources.mjs [url]   (needs google-chrome; start the server first, e.g. --port 8450)
import { spawn } from 'node:child_process';
const url = process.argv[2] || 'https://127.0.0.1:8450/', port = 9334;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const chrome = spawn('google-chrome', ['--headless=new', '--no-sandbox', '--ignore-certificate-errors', `--remote-debugging-port=${port}`,
  '--user-data-dir=' + (process.env.TMPDIR || '/tmp') + '/chrome_input_test', 'about:blank'], { stdio: 'ignore' });
let tabs; for (let i = 0; i < 60; i++) { try { tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(250); } }
const ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
let id = 0; const pend = {}; const errs = [];
ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.id && pend[m.id]) { pend[m.id](m); delete pend[m.id]; }
  if (m.method === 'Runtime.exceptionThrown') errs.push(JSON.stringify(m.params.exceptionDetails.exception?.description)); };
const send = (method, params = {}) => new Promise(r => { const i = ++id; pend[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async expr => { const r = await send('Runtime.evaluate', { returnByValue: true, awaitPromise: true, expression: expr });
  return r.result.result.value ?? r.result.exceptionDetails?.exception?.description; };
await send('Runtime.enable'); await send('Page.navigate', { url });
for (let i = 0; i < 80 && !(await ev('!!(window.dbg && window.dbg.resolveInputs)')); i++) await sleep(250);
await sleep(1500);   // websocket open
const r = await ev(`(() => {
    const D = window.dbg, out = {};
    const sent = []; const realWs = WebSocket.prototype.send;
    WebSocket.prototype.send = function (d) { if (typeof d === 'string') sent.push(JSON.parse(d)); return realWs.call(this, d); };
    const btn = (v, p) => ({ value: v, pressed: p, touched: p });
    const gp = (trig, grip, a) => ({ buttons: [btn(trig, trig > 0.5), btn(grip, !!grip), btn(0, false), btn(0, false), btn(0, !!a), btn(0, false), btn(0, false)], axes: [0, 0, 0, 0] });
    const mk = (h, g, x) => ({ handedness: h, gamepad: g, hand: null, profiles: ['test'], targetRaySpace: { h, x } });
    const L = mk('left', gp(0, 0, 0), -0.2);
    let R = mk('right', gp(1, 1, 1), 0.2);
    const session = { inputSources: [L, R] };
    const frame = { session,
      getPose: (sp) => ({ transform: { matrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, sp.x,1.0,-0.3,1]) } }),
      getViewerPose: () => ({ transform: { position: { x: 0, y: 1.6, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } }) };
    const step = () => { D.resolveInputs(frame); D.updateControllers(frame); };
    const lastXr = () => [...sent].reverse().find(m => m.t === 'xr');
    step(); step();
    out.held = { trig: lastXr()?.right?.trigger, grip: lastXr()?.right?.squeeze, cam: D.wristPanels.right.visible };
    // Quest swaps the right source: the old object stays in our hands with frozen pressed values, the new one is released
    const R2 = mk('right', gp(0, 0, 0), 0.2);
    session.inputSources = [L, R2];
    step(); step();
    out.swapped = { trig: lastXr()?.right?.trigger, grip: lastXr()?.right?.squeeze, cam: D.wristPanels.right.visible,
                    usesNew: D.handSrc.right === R2 };
    // right controller gone entirely
    session.inputSources = [L]; step();
    out.gone = { right: 'right' in lastXr(), cam: D.wristPanels.right.visible };
    // an exception in the right hand must not stop the left hand / the send
    const bad = mk('right', { get buttons() { throw new Error('boom'); }, axes: [0, 0, 0, 0] }, 0.2);
    session.inputSources = [L, bad];
    L.gamepad.buttons[1] = btn(1, true);
    const n0 = sent.filter(m => m.t === 'xr').length; step();
    out.isolated = { sentAgain: sent.filter(m => m.t === 'xr').length > n0, leftGrip: lastXr()?.left?.squeeze,
                     logged: sent.some(m => m.t === 'log' && /ERROR in controller right/.test(m.msg)) };
    out.logs = sent.filter(m => m.t === 'log').map(m => m.msg);
    return out;
})()`);
console.log(JSON.stringify(r, null, 1));
const ok = r.held.trig === 1 && r.held.grip === true && r.held.cam === true
    && r.swapped.trig === 0 && r.swapped.grip === false && r.swapped.cam === false && r.swapped.usesNew
    && r.gone.right === false && r.gone.cam === false
    && r.isolated.sentAgain && r.isolated.leftGrip === true && r.isolated.logged;
console.log(errs.length ? 'page errors: ' + errs.join(' | ') : 'no page errors');
console.log(ok ? 'PASS' : 'FAIL');
ws.close(); chrome.kill(); process.exit(ok ? 0 : 1);
