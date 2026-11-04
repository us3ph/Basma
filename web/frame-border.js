// Frame Border: WebGL port of the React Bits Pro "Frame Border" (3D & Shaders) using its documented props and
// defaults. A crisp line on a rounded rect (signed-distance, anti-aliased at any size), an inner glow that fades
// to zero toward the interior, two colours blended in OKLab whose gradient slowly travels around the frame,
// flowing noise through the glow and along the line, an intro that draws the line in from the top and then fades
// the glow in, and soft idle swells of light drifting along the frame. Click pulses / hover speed-up exist behind
// `interactive` (off here).
const DEFAULTS = { colors: ["#5227FF", "#FF9FFC"], radius: 16, lineWidth: 1, inset: 0, glow: 0.5, glowWidth: 110, noise: 0.6,
  noiseScale: 3, flowSpeed: 1, pulseSpeed: 1, idlePulses: 3, idleSpeed: 1, idleStrength: 1, intensity: 1, opacity: 1,
  grain: 0, interactive: true, intro: true, paused: false, dpr: 2 };

const VERT = `attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;
const FRAG = `
precision highp float;
uniform vec2 uRes; uniform float uDpr, uTime, uRadius, uLine, uInset, uGlow, uGlowW, uNoise, uNoiseScale, uFlow,
  uIdleN, uIdleSpeed, uIdleStr, uIntensity, uOpacity, uIntro, uGlowIn, uGrain;
uniform vec3 uLab0, uLab1;
uniform vec4 uPulse[4];            // (u, progress, active, _)
const float PI = 3.14159265359;

float sdRoundBox(vec2 p, vec2 b, float r){ vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }

// arc-length helpers (clockwise from the top centre); P = perimeter
float sHoriz(vec2 q, float a, float b, float r, float P){ return q.y < 0.0 ? (q.x >= 0.0 ? q.x : P + q.x) : a + PI*r + 2.0*b + (a - q.x); }
float sVert(vec2 q, float a, float b, float r){ return q.x >= 0.0 ? a + 0.5*PI*r + (q.y + b) : 3.0*a + 1.5*PI*r + 2.0*b + (b - q.y); }
float sCorner(vec2 v, float a, float b, float r){
  float ang = atan(v.y, v.x);
  if (v.x > 0.0 && v.y < 0.0) return a + r*(ang + 0.5*PI);
  if (v.x > 0.0) return a + 0.5*PI*r + 2.0*b + r*ang;
  if (v.y > 0.0) return 3.0*a + PI*r + 2.0*b + r*(ang - 0.5*PI);
  return 3.0*a + 1.5*PI*r + 4.0*b + r*(ang + PI);
}
// perimeter position of a point. Deep inside the frame two edges compete (top/bottom vs side); instead of a hard
// switch along the corner diagonals (which shows as a line in the glow) both are returned with a smooth weight.
void perim(vec2 q, float hx, float hy, float r, out float uA, out float uB, out float wA){
  float a = hx - r, b = hy - r, P = 4.0*a + 4.0*b + 2.0*PI*r;
  vec2 k = clamp(q, vec2(-a, -b), vec2(a, b)); vec2 v = q - k;
  bool onX = abs(v.x) < 1e-4, onY = abs(v.y) < 1e-4;
  wA = 1.0;
  if (onX && onY) {
    uA = sVert(q, a, b, r) / P; uB = sHoriz(q, a, b, r, P) / P;
    float diff = (hy - abs(q.y)) - (hx - abs(q.x));            // > 0: the side edge is nearer
    wA = smoothstep(-42.0, 42.0, diff);
  } else if (!onX && !onY) { uA = sCorner(v, a, b, r) / P; uB = uA; }
  else if (onX) { uA = sHoriz(q, a, b, r, P) / P; uB = uA; }
  else { uA = sVert(q, a, b, r) / P; uB = uA; }
  uA = fract(uA); uB = fract(uB);
}
float wd(float x, float y){ float d = abs(x - y); return min(d, 1.0 - d); }
float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0 - 2.0*f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y); }
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 7.1; a *= 0.5; } return s; }
vec3 oklabToLinear(vec3 c){
  float l_ = c.x + 0.3963377774*c.y + 0.2158037573*c.z, m_ = c.x - 0.1055613458*c.y - 0.0638541728*c.z, s_ = c.x - 0.0894841775*c.y - 1.2914855480*c.z;
  float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
  return vec3(4.0767416621*l - 3.3077115913*m + 0.2309699292*s, -1.2684380046*l + 2.6097574011*m - 0.3413193965*s, -0.0041960863*l - 0.7034186147*m + 1.7076147010*s);
}
vec3 encode(vec3 c){ c = clamp(c, 0.0, 1.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c)); }

void field(float u, float depth, float t, out vec3 col, out float lit, out float sw){
  float blend = 0.5 + 0.5 * sin(2.0*PI*(u - t * 0.025 * uFlow));         // gradient travelling around the frame
  col = oklabToLinear(mix(uLab0, uLab1, blend));
  vec2 ring = vec2(cos(2.0*PI*u), sin(2.0*PI*u)) * uNoiseScale;         // seamless around the loop
  float n = fbm(ring + vec2(depth * uNoiseScale * 0.8 + t * 0.12 * uFlow, -t * 0.08 * uFlow));
  lit = clamp(1.0 + uNoise * (n - 0.5) * 1.6, 0.0, 2.0);
  sw = 0.0;
  for (int i = 0; i < 8; i++) { if (float(i) >= uIdleN) break;           // idle swells drifting along the frame
    float fi = float(i);
    float pos = fract(t * 0.03 * uIdleSpeed * (1.0 + 0.23*fi) + fi / max(uIdleN, 1.0));
    float x = wd(u, pos) / 0.055; sw += exp(-x*x) * uIdleStr; }
  for (int i = 0; i < 4; i++) { vec4 pl = uPulse[i]; if (pl.z < 0.5) continue;
    float a1 = fract(pl.x + pl.y * 0.5), b1 = fract(pl.x - pl.y * 0.5 + 1.0);
    float x1 = wd(u, a1) / 0.03, x2 = wd(u, b1) / 0.03; sw += 1.6 * (exp(-x1*x1) + exp(-x2*x2)) * (1.0 - pl.y); }
}

void main(){
  vec2 p = vec2(gl_FragCoord.x, uRes.y * uDpr - gl_FragCoord.y) / uDpr;
  vec2 half_ = uRes * 0.5 - vec2(uInset + 0.5*uLine);
  float r = max(0.0, min(uRadius - uInset - 0.5*uLine, min(half_.x, half_.y)));
  vec2 q = p - uRes * 0.5;
  float d = sdRoundBox(q, half_, r);                     // <0 inside the line
  float t = uTime;
  // glow depth: smooth-min of the distances to the two nearest edges, so the falloff curves around the corners
  // instead of forming a ridge along the diagonals (the exact box distance has a kink there)
  float dxE = half_.x - abs(q.x), dyE = half_.y - abs(q.y), kk = 60.0;
  float hh = clamp(0.5 + 0.5 * (dyE - dxE) / kk, 0.0, 1.0);
  float smE = mix(dyE, dxE, hh) - kk * hh * (1.0 - hh);
  float depth = clamp(min(-d, smE) / uGlowW, 0.0, 1.0);
  float uA, uB, wA; perim(q, half_.x, half_.y, r, uA, uB, wA);
  vec3 col, cB; float lit, lB, sw, sB;
  field(uA, depth, t, col, lit, sw);
  if (wA < 0.999) { field(uB, depth, t, cB, lB, sB); col = mix(cB, col, wA); lit = mix(lB, lit, wA); sw = mix(sB, sw, wA); }
  float u = wA >= 0.5 ? uA : uB;

  // intro: the line is drawn in from the top centre in both directions, then the glow fades in
  float drawn = uIntro >= 1.0 ? 1.0 : smoothstep(0.0, 0.004, uIntro * 0.502 - wd(u, 0.0));   // closes fully at the bottom

  float aa = 0.75 / uDpr + 0.25;
  float line = (1.0 - smoothstep(0.5*uLine - aa, 0.5*uLine + aa, abs(d))) * drawn;
  float fall = pow(1.0 - depth, 2.6) * step(d, 0.0);
  float glowA = uGlow * fall * (0.25 + 0.55 * lit + 0.9 * sw) * uGlowIn;
  float lineA = line * clamp(0.55 + 0.35 * lit + 0.8 * sw, 0.0, 1.6);
  vec3 lineCol = mix(col, vec3(1.0), clamp(0.18 * sw + 0.08 * (lit - 1.0), 0.0, 0.45));

  vec3 rgb = col * glowA + lineCol * lineA;
  float alpha = clamp((glowA + lineA) * uIntensity, 0.0, 1.0);
  vec3 c = rgb / max(glowA + lineA, 1e-4);
  if (uGrain > 0.0) c += (hash(p + fract(t)) - 0.5) * uGrain * 0.2;
  alpha *= uOpacity;
  gl_FragColor = vec4(encode(c) * alpha, alpha);
}`;

function hexToOklab(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  const lin = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
  const [r, g, b] = lin;
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
          0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}

export function frameBorder(root, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  if (getComputedStyle(root).position === "static") root.style.position = "relative";
  root.style.borderRadius = `${o.radius}px`;
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", pointerEvents: "none", zIndex: "1", borderRadius: `${o.radius}px` });
  root.prepend(canvas);
  const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: false });
  if (!gl) { root.style.boxShadow = `inset 0 0 0 ${o.lineWidth}px ${o.colors[1]}`; return { pulse() {}, replay() {} }; }

  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const U = (n) => gl.getUniformLocation(prog, n);
  const set1 = (n, v) => gl.uniform1f(U(n), v);
  gl.uniform3fv(U("uLab0"), hexToOklab(o.colors[0])); gl.uniform3fv(U("uLab1"), hexToOklab(o.colors[1]));
  set1("uRadius", o.radius); set1("uLine", o.lineWidth); set1("uInset", o.inset); set1("uGlow", o.glow); set1("uGlowW", o.glowWidth);
  set1("uNoise", o.noise); set1("uNoiseScale", o.noiseScale); set1("uFlow", o.flowSpeed); set1("uIdleN", o.idlePulses);
  set1("uIdleSpeed", o.idleSpeed); set1("uIdleStr", o.idleStrength); set1("uIntensity", o.intensity); set1("uOpacity", o.opacity); set1("uGrain", o.grain);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches || o.paused;
  let W = 0, H = 0, dpr = 1, t0 = performance.now(), last = t0, clock = 0, hover = 1, raf = 0;
  let pulses = [];
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, o.dpr);
    const r = root.getBoundingClientRect(); W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  function draw(now) {
    const dt = Math.max(0, Math.min(0.05, (now - last) / 1000)); last = now;   // never step backwards
    clock += reduced ? 0 : dt * hover;
    const since = Math.max(0, (now - t0) / 1000);
    const intro = !o.intro || reduced ? 1 : Math.min(1, since / 1.2);
    const glowIn = !o.intro || reduced ? 1 : Math.max(0, Math.min(1, (since - 1.0) / 1.0));
    gl.uniform2f(U("uRes"), W, H); set1("uDpr", dpr); set1("uTime", clock + 17.0); set1("uIntro", intro); set1("uGlowIn", glowIn);
    pulses.forEach(pl => { pl.p += dt * 0.6 * o.pulseSpeed; }); pulses = pulses.filter(pl => pl.p < 1);
    const pv = new Float32Array(16); pulses.slice(-4).forEach((pl, i) => { pv.set([pl.u, pl.p, 1, 0], i * 4); });
    gl.uniform4fv(U("uPulse[0]"), pv);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLES, 0, 6);
    raf = requestAnimationFrame(draw);
  }
  function pulse(x, y) {
    const r = root.getBoundingClientRect(); let u = 0;
    if (x !== undefined) {       // approximate perimeter position from the click direction (clockwise from top centre)
      const qx = x - r.left - r.width / 2, qy = y - r.top - r.height / 2;
      u = (Math.atan2(qx, -qy) / (2 * Math.PI) + 1) % 1;
    }
    pulses.push({ u, p: 0 });
  }
  if (o.interactive) {
    root.addEventListener("click", (e) => { if (!reduced) pulse(e.clientX, e.clientY); });
    root.addEventListener("mouseenter", () => { hover = 1.6; }); root.addEventListener("mouseleave", () => { hover = 1; });
  }
  new ResizeObserver(resize).observe(root); resize();
  document.addEventListener("visibilitychange", () => { cancelAnimationFrame(raf); if (!document.hidden) { last = performance.now(); raf = requestAnimationFrame(draw); } });
  raf = requestAnimationFrame(draw);
  return { pulse, replay: () => { t0 = performance.now(); } };
}
