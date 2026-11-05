// Ghost Type: vanilla port of the React Bits Pro "Ghost Type" headline (a headline that writes itself with AI-style
// ghost completions), following its documented behaviour, props and defaults: the base text types in, then after a
// think pause a faded suggestion streams in word by word with a "Tab" key hint, is accepted (the accent sweeps
// through it), holds, is cleared by select-and-delete (or backspace), and the next suggestion follows. Hover pauses
// the timing, click accepts the current suggestion or skips ahead. The text stays left-aligned inside a box sized
// for the longest sentence, so nothing around the headline moves while it types.
const DEFAULTS = { text: "Build anything", completions: [" with a single prompt."], accentColor: "#8B5CF6", accentColors: null,
  ghostOpacity: 0.38, align: "center", typeSpeed: 45, streamSpeed: 70, streamBy: "word", thinkDelay: 600, acceptDelay: 1200,
  holdDelay: 1800, clearStyle: "select", caret: "bar", caretBlink: true, hint: true, hintLabel: "Tab", sweep: true, loop: true,
  startOnView: true, pauseOnHover: true, clickToAccept: true, paused: false, onAccept: null, fill: null, acceptedFill: null };

const CSS = `
.gt { position: relative; text-align: var(--gt-align, center); }
.gt-box { display: inline-block; max-width: 100%; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: break-word; }
.gt-measure { position: absolute; left: 0; top: 0; visibility: hidden; pointer-events: none; }
.gt-fill { color: var(--gt-color, currentColor); -webkit-box-decoration-break: clone; box-decoration-break: clone; }
.gt.gt-filled .gt-fill { background-image: var(--gt-bg); -webkit-background-clip: text; background-clip: text; color: transparent; }
/* accepted endings keep their own fill; slice = one gradient running through the ending across line breaks */
.gt.gt-accfill .gt-done > .gt-fill { background-image: var(--gt-acc); -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-box-decoration-break: slice; box-decoration-break: slice; }
.gt-ghost { opacity: var(--gt-ghost, .38); }
.gt-done { border-radius: .08em; transition: background-color .18s ease; -webkit-box-decoration-break: clone; box-decoration-break: clone; }
.gt-done.sel { background-color: var(--gt-sel); }
.gt-done.fresh > .gt-fill { animation: gt-solid .35s ease forwards; }
@keyframes gt-solid { from { opacity: var(--gt-ghost, .38); } to { opacity: 1; } }
.gt .gt-done.sweep > .gt-fill { background-image: linear-gradient(90deg, transparent 0%, transparent 38%, var(--gt-a1) 46%, var(--gt-a2) 54%, transparent 62%, transparent 100%), var(--gt-acc, var(--gt-bg, linear-gradient(currentColor, currentColor)));
  background-size: 320% 100%, 100% 100%; background-repeat: no-repeat; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: gt-solid .35s ease forwards, gt-sweep .8s cubic-bezier(.4,.1,.3,1) forwards; }
@keyframes gt-sweep { from { background-position: 100% 0, 0 0; } to { background-position: 0% 0, 0 0; } }
.gt-caret { display: inline-block; vertical-align: -.1em; background: var(--gt-a2); border-radius: .02em; margin-left: .03em; margin-right: -.03em; }
.gt-caret.bar { width: .055em; height: .98em; }
.gt-caret.block { width: .5em; height: .98em; opacity: .55; margin-right: -.47em; }
.gt-caret.underscore { width: .5em; height: .07em; vertical-align: -.06em; margin-right: -.47em; }
.gt-caret.blink { animation: gt-blink 1.06s steps(1, end) infinite; }
@keyframes gt-blink { 0% { opacity: 1; } 50% { opacity: 0; } }
.gt-caret.block.blink { animation-name: gt-blink-block; }
@keyframes gt-blink-block { 0% { opacity: .55; } 50% { opacity: 0; } }
.gt-hint { display: inline-block; vertical-align: .55em; margin-left: .55em; padding: .32em .62em; font-size: .24em; font-weight: 600;
  line-height: 1; letter-spacing: .02em; border-radius: .5em; border: 1px solid rgba(255,255,255,.28); background: rgba(255,255,255,.06);
  color: var(--gt-color, currentColor); opacity: 0; transform: translateY(.2em); transition: opacity .25s ease, transform .25s ease; white-space: nowrap; }
.gt-hint.on { opacity: .75; transform: none; }
`;

let cssAdded = false;
function addCSS() { if (cssAdded) return; cssAdded = true; const s = document.createElement("style"); s.textContent = CSS; document.head.appendChild(s); }

const span = (cls, text = "") => { const s = document.createElement("span"); s.className = cls; s.textContent = text; return s; };

export function ghostType(el, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  addCSS();
  const [a1, a2] = o.accentColors || [o.accentColor, o.accentColor];
  el.classList.add("gt");
  el.style.setProperty("--gt-align", o.align);
  el.style.setProperty("--gt-ghost", String(o.ghostOpacity));
  el.style.setProperty("--gt-a1", a1); el.style.setProperty("--gt-a2", a2);
  if (o.fill) { el.classList.add("gt-filled"); el.style.setProperty("--gt-bg", o.fill); }          // gradient text instead of a flat colour
  if (o.acceptedFill) { el.classList.add("gt-accfill"); el.style.setProperty("--gt-acc", o.acceptedFill); }
  el.style.setProperty("--gt-sel", `color-mix(in srgb, ${a2} 30%, transparent)`);
  if (!el.hasAttribute("aria-label")) el.setAttribute("aria-label", o.text);

  // live line: base, accepted ending, caret, ghost suggestion, key hint
  const box = span("gt-box"); box.setAttribute("aria-hidden", "true");
  const base = span("gt-fill"), done = span("gt-done"), doneFill = span("gt-fill"), caret = span(`gt-caret ${o.caret}`);
  const ghost = span("gt-fill gt-ghost"), hint = span("gt-hint", o.hintLabel);
  done.appendChild(doneFill); box.append(base, done, caret, ghost); if (o.hint) box.appendChild(hint);
  hint.style.display = "none";
  const measure = span("gt-box gt-measure"); measure.setAttribute("aria-hidden", "true");
  el.replaceChildren(box, measure);

  // reserve the size of the longest sentence (with caret and hint) so the layout never shifts
  function reserve() {
    box.style.width = ""; box.style.minHeight = "";
    let w = 0, h = 0;
    for (const c of o.completions.length ? o.completions : [""]) {
      measure.replaceChildren(span("gt-fill", o.text + c), span(`gt-caret ${o.caret}`));
      if (o.hint && c) { const k = span("gt-hint on", o.hintLabel); measure.appendChild(k); }
      const r = measure.getBoundingClientRect(); w = Math.max(w, r.width); h = Math.max(h, r.height);
    }
    measure.replaceChildren();
    box.style.width = `${Math.ceil(w) + 1}px`; box.style.minHeight = `${Math.ceil(h)}px`;
  }

  // timing: waits only count while not paused (prop, hover, hidden tab); a click or restart cancels the running sequence
  let token = 0, hover = false, state = "idle", idx = 0, started = false;
  const CANCEL = Symbol("cancel");
  const halted = () => o.paused || (o.pauseOnHover && hover);
  function sleep(ms, t) {
    return new Promise((res, rej) => {
      let left = ms, last = performance.now();
      const tick = (now) => {
        if (t !== token) return rej(CANCEL);
        if (!halted()) left -= now - last;
        last = now;
        if (left <= 0) res(); else requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }
  const jitter = (ms) => ms * (0.6 + Math.random() * 0.8);
  const blink = (on) => caret.classList.toggle("blink", !!(on && o.caretBlink));
  const showHint = (on) => {
    if (!o.hint) return;
    if (on) { hint.style.display = ""; requestAnimationFrame(() => hint.classList.add("on")); }
    else { hint.classList.remove("on"); hint.style.display = "none"; }
  };
  const tokens = (c) => (o.streamBy === "char" ? [...c] : c.match(/\s*\S+\s*/g) || [c]);

  function acceptNow(i) {
    const c = o.completions[i];
    ghost.textContent = ""; showHint(false);
    doneFill.textContent = c; done.classList.remove("sel", "sweep", "fresh"); void done.offsetWidth;
    done.classList.add(o.sweep ? "sweep" : "fresh");
    o.onAccept?.(c, i);
  }
  function clearNow() { doneFill.textContent = ""; ghost.textContent = ""; done.classList.remove("sel", "sweep", "fresh"); showHint(false); }

  async function run(from, i, t) {
    try {
      const n = o.completions.length;
      if (from === "type") {
        state = "type"; blink(false);
        for (const ch of o.text.slice(base.textContent.length)) { base.textContent += ch; await sleep(jitter(o.typeSpeed), t); }
        from = "think";
      }
      if (!n) { state = "idle"; blink(true); return; }
      while (true) {
        idx = i;
        if (from === "think") { state = "think"; blink(true); await sleep(o.thinkDelay, t); from = "stream"; }
        if (from === "stream") {
          state = "stream"; blink(false); ghost.textContent = "";
          const toks = tokens(o.completions[i]);
          for (let k = 0; k < toks.length; k++) {
            ghost.textContent += toks[k]; if (k === 0) showHint(true);
            await sleep(o.streamBy === "char" ? o.streamSpeed / 3 : o.streamSpeed, t);
          }
          state = "await"; blink(true); await sleep(o.acceptDelay, t);
          acceptNow(i); from = "hold";
        }
        if (from === "hold") {
          state = "hold"; blink(true);
          if (!o.loop && i === n - 1) { state = "idle"; return; }
          await sleep(o.holdDelay, t);
          state = "clear"; blink(false);
          if (o.clearStyle === "backspace") {
            while (doneFill.textContent.length) { doneFill.textContent = doneFill.textContent.slice(0, -1); await sleep(jitter(o.typeSpeed * 0.6), t); }
          } else { done.classList.add("sel"); await sleep(420, t); }
          clearNow(); i = (i + 1) % n; from = "think";
        }
      }
    } catch (e) { if (e !== CANCEL) throw e; }
  }
  const go = (from, i) => { token++; run(from, i, token); };

  // click: accept the suggestion on screen, otherwise skip ahead to the next one
  function accept() {
    const n = o.completions.length; if (!n) return;
    if (state === "type") { base.textContent = o.text; go("stream", idx); return; }
    if (state === "think") { go("stream", idx); return; }
    if (state === "stream" || state === "await") { token++; acceptNow(idx); go("hold", idx); return; }
    next();
  }
  function next() {
    const n = o.completions.length; if (!n) return;
    base.textContent = o.text; clearNow(); go("stream", (idx + 1) % n);
  }
  function restart() { base.textContent = ""; clearNow(); idx = 0; go("type", 0); }

  if (o.clickToAccept) el.addEventListener("click", accept);
  el.addEventListener("pointerenter", () => { hover = true; });
  el.addEventListener("pointerleave", () => { hover = false; });

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function start() {
    if (started) return; started = true;
    if (reduced) { base.textContent = o.text; if (o.completions.length) { doneFill.textContent = o.completions[0]; } blink(false); return; }
    go("type", 0);
  }
  reserve();
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(reserve, 150); });
  document.fonts?.ready.then(reserve); document.fonts?.addEventListener?.("loadingdone", reserve);   // web fonts change the size
  if (o.startOnView && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); start(); } });
    io.observe(el);
  } else start();

  return { accept, next, restart, setPaused(p) { o.paused = !!p; }, destroy() { token++; el.replaceChildren(); } };
}
