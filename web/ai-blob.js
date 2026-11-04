// AI Blob: WebGL port of the React Bits Pro "AI Blob" (a glowing glass orb that listens, thinks and speaks),
// following its documented look and props: colours flowing around a bright rim, a glowing core, white glints on
// the glass, a soft halo; states idle / listening (brighter rim and halo) / thinking (colours spin, core pulses) /
// speaking (swells with a simulated voice level); tap squish + flare; spring-in intro.
const DEFAULTS = { size: 44, colors: ["#ff006e", "#8338ec", "#3a86ff", "#06ffa5"], state: "idle", reactivity: 1, animationSpeed: 1,
  colorSpeed: 1, glowIntensity: 0.8, noiseScale: 3, innerScale: 1, coreIntensity: 1, rimWidth: 1, rimIntensity: 1, glints: 1,
  halo: 1, wobble: 0, intro: true, paused: false, interactive: true, canvasScale: 2.2, dpr: 2 };

const VERT = `attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;
const FRAG = `
precision highp float;
uniform vec2 uRes; uniform float uDpr, uT, uR, uN, uSpin, uCore, uRim, uHalo, uGlints, uGlow, uSqX, uSqY, uFlare, uVol, uScale,
  uNoiseScale, uRimW, uInner, uWobble;
uniform vec3 uC0, uC1, uC2, uC3;
const float PI = 3.14159265359;
float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0 - 2.0*f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y); }
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 7.1; a *= 0.5; } return s; }
vec3 getC(float i){ return i < 0.5 ? uC0 : (i < 1.5 ? uC1 : (i < 2.5 ? uC2 : uC3)); }
vec3 pal(float x){ x = fract(x) * uN; float i = floor(x); float f = smoothstep(0.0, 1.0, fract(x));
  return mix(getC(i), getC(mod(i + 1.0, uN)), f); }
mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
vec3 encode(vec3 c){ c = clamp(c, 0.0, 1.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c)); }
void main(){
  vec2 fc = vec2(gl_FragCoord.x, uRes.y * uDpr - gl_FragCoord.y) / uDpr;
  vec2 p = (fc - uRes * 0.5) / max(uR * uScale, 1e-3);
  p /= vec2(uSqX, uSqY);
  float rr = length(p), a = atan(p.y, p.x + 1e-6);
  vec2 ring2 = vec2(cos(a), sin(a));
  float sh = fbm(ring2 * uNoiseScale + vec2(uT * 0.45, -uT * 0.32));
  float rimR = 1.0 + (sh - 0.5) * 0.05 + uVol * 0.09 + uWobble * 0.06 * sin(a * 3.0 + uT * 1.7);
  float colPos = a / (2.0 * PI) + uSpin;
  vec3 rimCol = pal(colPos);
  float ringW = 0.07 * uRimW;
  float rx = (rr - rimR) / ringW;                                        // x*x, never pow(): pow(negative, 2.0) is undefined in GLSL
  float ring = exp(-rx * rx) * (0.75 + 0.5 * sh);
  float inside = 1.0 - smoothstep(rimR - 0.02, rimR + 0.02, rr);
  vec2 sp = rot(uT * 0.35) * p;
  float sw = fbm(sp * 1.7 + vec2(uT * 0.25, uT * 0.18));
  vec3 innerCol = pal(colPos + sw * 0.9 + 0.25);                         // whole turns only: no seam at +-180 deg
  float fres = pow(clamp(rr / rimR, 0.0, 1.0), 3.0);
  vec3 glass = innerCol * (0.10 + 0.50 * fres) * inside;
  float core = exp(-rr * rr / (0.20 * uInner * uInner)) * uCore;
  vec3 coreCol = mix(pal(uSpin + 0.5), vec3(1.0), 0.3);
  vec2 q1 = (p - vec2(-0.36, -0.44)) * vec2(1.0, 1.9); float g1 = exp(-dot(q1, q1) / 0.025);
  vec2 q2 = p - vec2(0.44, 0.38); float g2 = exp(-dot(q2, q2) / 0.003);
  float glint = (0.85 * g1 + 0.6 * g2) * uGlints * inside;
  float outD = max(rr - rimR, 0.0);
  float edgeR = 0.5 * min(uRes.x, uRes.y) / max(uR * uScale, 1e-3);      // canvas edge in orb radii
  float halo = exp(-outD / (0.30 * max(uHalo, 0.01))) * (1.0 - inside) * 0.6
             * (1.0 - smoothstep(0.55 * edgeR, 0.92 * edgeR, rr));          // reaches zero before the canvas edge
  vec3 col = glass + coreCol * core * inside * 0.85 + rimCol * ring * uRim * 1.25 + rimCol * halo * uRim * uHalo + vec3(1.0) * glint;
  col *= uGlow * (1.0 + uFlare);
  vec3 e = encode(col);
  float alpha = clamp(max(max(e.r, e.g), e.b), 0.0, 1.0);
  gl_FragColor = vec4(e, alpha);
}`;

const toLin = (hex) => { const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); };

export function aiBlob(host, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const S = o.size * o.canvasScale;                       // canvas larger than the orb so the halo can spill out
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { position: "absolute", left: "50%", top: "50%", width: `${S}px`, height: `${S}px`,
    transform: "translate(-50%, -50%)", pointerEvents: "none", zIndex: "0" });
  host.prepend(canvas);
  const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: false });
  if (!gl) { host.style.background = o.colors[0]; return { setState() {}, tap() {} }; }
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const U = (n) => gl.getUniformLocation(prog, n); const set1 = (n, v) => gl.uniform1f(U(n), v);
  const cols = o.colors.slice(0, 4).map(toLin); while (cols.length < 4) cols.push(cols[cols.length - 1]);
  ["uC0", "uC1", "uC2", "uC3"].forEach((n, i) => gl.uniform3fv(U(n), cols[i]));
  set1("uN", Math.max(2, Math.min(4, o.colors.length))); set1("uNoiseScale", o.noiseScale); set1("uRimW", o.rimWidth);
  set1("uInner", o.innerScale); set1("uWobble", o.wobble); set1("uGlints", o.glints);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches || o.paused;
  let state = o.state, dpr = 1, last = performance.now(), born = last, clock = 0, spin = 0, tapAt = -1e9, speakUntil = 0;
  const cur = { spin: 0.05, core: 0.8, rim: 1, halo: 1, vol: 0 };
  function resize() { dpr = Math.min(window.devicePixelRatio || 1, o.dpr); canvas.width = Math.round(S * dpr); canvas.height = Math.round(S * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height); }
  function targets(now) {
    const t = clock;
    if (state === "thinking") return { spin: 0.55 * o.colorSpeed, core: o.coreIntensity * (0.75 + 0.45 * Math.sin(t * 5.5)), rim: 1.15, halo: 1.05, vol: 0 };
    if (state === "listening") return { spin: 0.08 * o.colorSpeed, core: o.coreIntensity * 0.95, rim: 1.2, halo: 1.12, vol: 0 };
    if (state === "speaking") {                                     // simulated voice level (autoVolume)
      const v = Math.max(0, 0.55 * Math.sin(t * 9.0) * Math.sin(t * 3.1 + 1.0) + 0.35 * Math.sin(t * 15.0 + 2.0) * Math.sin(t * 1.3));
      return { spin: 0.12 * o.colorSpeed, core: o.coreIntensity * (0.9 + 0.6 * v * o.reactivity), rim: 1.2 + 0.5 * v * o.reactivity, halo: 1.1 + 0.4 * v, vol: v * o.reactivity };
    }
    return { spin: 0.05 * o.colorSpeed, core: o.coreIntensity * (0.78 + 0.12 * Math.sin(t * 1.3)), rim: 1, halo: 1, vol: 0 };   // idle: gentle breathing
  }
  function draw(now) {
    const dt = Math.max(0, Math.min(0.05, (now - last) / 1000)); last = now;   // rAF time can precede creation: never step backwards
    if (!reduced) clock += dt * o.animationSpeed;
    if (state === "speaking" && now > speakUntil) state = hostState;
    const tg = targets(now), k = 1 - Math.exp(-dt * 6);
    for (const key of Object.keys(cur)) cur[key] += (tg[key] - cur[key]) * k;
    if (!reduced) spin += cur.spin * dt;
    const ta = (now - tapAt) / 1000;                                 // tap: squish (damped spring) + flare
    const sq = ta < 1.2 ? 0.14 * Math.exp(-7 * ta) * Math.cos(20 * ta) : 0;
    const flare = ta < 1.2 ? 0.7 * Math.exp(-5 * ta) : 0;
    const tb = (now - born) / 1000;                                   // intro: spring-in
    const scale = !o.intro || reduced ? 1 : Math.max(0, 1 - Math.exp(-7 * tb) * Math.cos(9 * tb));
    gl.uniform2f(U("uRes"), S, S); set1("uDpr", dpr); set1("uT", clock); set1("uR", o.size / 2 - 2); set1("uSpin", spin);
    set1("uCore", cur.core); set1("uRim", cur.rim * o.rimIntensity); set1("uHalo", cur.halo * o.halo); set1("uGlow", o.glowIntensity);
    set1("uSqX", 1 + sq); set1("uSqY", 1 - sq); set1("uFlare", flare); set1("uVol", cur.vol); set1("uScale", scale);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(draw);
  }
  let hostState = state;
  resize(); requestAnimationFrame(draw);
  if (o.interactive) host.addEventListener("pointerdown", () => { tapAt = performance.now(); });
  return {
    setState(s) { hostState = s; if (state !== "speaking") state = s; },
    speak(ms = 1600) { state = "speaking"; speakUntil = performance.now() + ms; },
    tap() { tapAt = performance.now(); },
    replay() { born = performance.now(); },          // restart the spring-in intro (used when the button is revealed)
    get state() { return state; },
  };
}
