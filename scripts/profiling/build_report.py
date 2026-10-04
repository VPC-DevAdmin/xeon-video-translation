#!/usr/bin/env python3
"""Render the profiling capture into an HTML report (Chart.js timeline + tables)."""
import json, sys, html
from pathlib import Path

RUN = Path(sys.argv[1]); OUT = Path(sys.argv[2])
R = json.loads((RUN / "report.json").read_text())
T = R["timeline"]

GPU_ROLE = {"0": "backend: whisper, XTTS, mux", "1": "vLLM Qwen3-30B (translate, review)", "2": "avatar renderer + WebRTC codecs",
            "3": "lipsync-fast worker A (+coordinator)", "4": "batch coordinator + worker", "5": "lipsync-fast worker B",
            "6": "batch worker", "7": "batch worker"}

def stage_rows(mode):
    rows = []
    for s in R["modes"][mode]["stages"]:
        busy = sorted(((i, v["utilization.gpu"][0], v["utilization.gpu"][1], v["memory.used"][1] / 1024) for i, v in s["gpu"].items()
                       if v.get("utilization.gpu") and v["utilization.gpu"][0] >= 3), key=lambda x: -x[1])
        gpus = ", ".join(f"GPU {i}: {m:.0f}% (peak {p:.0f}%, {mem:.0f} GB)" for i, m, p, mem in busy) or "none above 3%"
        cont = sorted(s["containers"].items(), key=lambda kv: -kv[1][0])[:2]
        cont_s = ", ".join(f"{n.replace('polyglot-', '').replace('xeon-video-translation-', '')} {m:.0f}% (peak {p:.0f}%)" for n, (m, p) in cont if m >= 1) or "idle"
        rows.append(f"<tr><td>{s['stage']}</td><td class=num>{s['seconds']:.1f}</td><td>{gpus}</td><td class=num>{s['host'].get('cpu_busy_pct', 0):.1f}%</td><td>{cont_s}</td></tr>")
    return "\n".join(rows)

def hot_rows(name, kind, n=8):
    h = R["pyspy"].get(name)
    if not h:
        return "<tr><td colspan=2>no capture</td></tr>"
    return "\n".join(f"<tr><td><code>{html.escape(k)}</code></td><td class=num>{v:.1f}%</td></tr>" for k, v in h[kind][:n])

bands = []
for mode, color in (("fast", "rgba(31,119,180,0.10)"), ("quality", "rgba(214,96,39,0.10)")):
    for s in R["modes"][mode]["stages"]:
        bands.append({"mode": mode, "stage": s["stage"], "start": s["start"], "end": s["end"], "color": color})

data = {"t": T["t"], "gpu": T["gpu_util"], "mem": T["gpu_mem_gb"], "host_t": T["host_t"], "host_cpu": T["host_cpu"], "bands": bands}
fast = R["modes"]["fast"]; qual = R["modes"]["quality"]
fast_total = sum(s["seconds"] for s in fast["stages"]); qual_total = sum(s["seconds"] for s in qual["stages"])

page = f"""<title>XE7740 Flow Profile</title>
<meta name="description" content="Stage-aligned GPU, CPU and function profile of one fast and one quality translation job on the XE7740.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column for findings, full-width bands for the timeline and tables. */
:root {{
  --bg: #f6f5f1; --fg: #1d2127; --muted: #5d6673; --rule: #d9d6cd; --panel: #ffffff;
  --accent: #1f5f8b; --fast: #1f77b4; --quality: #d66027; --warn: #b7791f; --good: #2e7d4f;
  --display: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #15181d; --fg: #e8e6e0; --muted: #a0a7b1; --rule: #2e343c; --panel: #1c2026; --accent: #7fb3d9; --fast: #5fa3d9; --quality: #ec8c5c; --warn: #d9a441; --good: #5fb88a; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #15181d; --fg: #e8e6e0; --muted: #a0a7b1; --rule: #2e343c; --panel: #1c2026; --accent: #7fb3d9; --fast: #5fa3d9; --quality: #ec8c5c; --warn: #d9a441; --good: #5fb88a; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--display); line-height: 1.5; margin: 0; padding-block: 32px 64px; padding-inline: 16px; }}
main {{ max-width: 1100px; margin: 0 auto; display: grid; gap: 36px; }}
h1 {{ font-size: clamp(1.6rem, 3vw, 2.2rem); margin: 0; text-wrap: balance; font-weight: 600; letter-spacing: -0.01em }}
h2 {{ font-size: 1.15rem; margin: 0 0 12px; font-weight: 600 }}
h3 {{ font-size: 0.95rem; margin: 18px 0 6px; font-weight: 600 }}
p, li {{ max-width: 70ch }}
.lede {{ color: var(--muted); margin: 6px 0 0 }}
.eyebrow {{ font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 500 }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px }}
.kpi {{ border-top: 2px solid var(--rule); padding-top: 8px }}
.kpi b {{ display: block; font-size: 1.5rem; font-variant-numeric: tabular-nums; font-weight: 600 }}
.kpi span {{ color: var(--muted); font-size: 0.85rem }}
.chart {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; }}
.chart canvas {{ max-width: 100%; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.8rem; color: var(--muted); margin-top: 8px }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle }}
.tablewrap {{ overflow-x: auto }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top }}
th {{ font-weight: 500; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase }}
td.num {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap }}
code {{ font-family: var(--mono); font-size: 0.82em; }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px }}
.cols > div {{ min-width: 0 }}
.finding {{ border-left: 3px solid var(--rule); padding: 2px 0 2px 14px; margin: 0 0 14px }}
.finding.binding {{ border-color: var(--quality) }}
.finding.cold {{ border-color: var(--warn) }}
.finding.ok {{ border-color: var(--good) }}
.finding b {{ font-weight: 600 }}
.tag {{ font-size: 0.72rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted) }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important }} }}
</style>
<main>
<header>
  <div class="eyebrow">XE7740 · 8× RTX PRO 6000 Blackwell · 4 Oct 2026 · commit a53c577</div>
  <h1>Where the translation pipeline spends its time</h1>
  <p class="lede">One fast job and one quality job on the 52 s 1080×1920 fixture, sampled every second across all eight GPUs, the host and every container, with py-spy on the renderer coordinator, a denoise worker and the backend. Warm models, face track cached.</p>
</header>

<section class="kpis">
  <div class="kpi"><b>{fast_total:.0f} s</b><span>fast job, {fast['stages'][-2]['seconds']:.0f} s of it in lipsync (10 steps, GPUs 3+5)</span></div>
  <div class="kpi"><b>{qual_total:.0f} s</b><span>quality job, {qual['stages'][-2]['seconds']:.0f} s in lipsync (40 steps, GPUs 4+6+7)</span></div>
  <div class="kpi"><b>63% / 48%</b><span>mean SM utilization of the two fast GPUs while rendering</span></div>
  <div class="kpi"><b>1.3%</b><span>host CPU busy (344 threads) during lipsync</span></div>
  <div class="kpi"><b>&lt;1%</b><span>NVENC / NVDEC engine utilization during rendering</span></div>
</section>

<section>
  <h2>Timeline: GPU utilization by card, with pipeline stages</h2>
  <div class="chart"><canvas id="util" height="130"></canvas>
  <div class="legend" id="legend"></div></div>
  <p class="lede">Blue bands are the fast job's stages, orange the quality job's. GPU 1 spikes are the LLM answering translation and review prompts. GPU 0 carries whisper, XTTS and the mux. Memory is included in the second chart.</p>
  <div class="chart"><canvas id="mem" height="110"></canvas></div>
</section>

<section>
  <h2>The binding constraint: one Python thread feeding the denoise workers</h2>
  <div class="finding binding"><b>The renderer coordinator is the bottleneck, not the GPUs.</b> During the fast job's lipsync the two render GPUs averaged 63% and 48% busy while the coordinator process ran one Python thread at full tilt. py-spy puts that thread's time in three places: encoding the masked frames into latents on the coordinator GPU (<code>prepare_mask_latents</code>, 25%), copying every chunk's five conditioning tensors from GPU to host memory to hand them to the worker processes (the generator at <code>lipsync_pipeline.py:1096</code>, 25%) and pasting finished faces back into the frames on the CPU (<code>restore_img</code>, 20%). The workers, meanwhile, spend 56% of their time in the DDIM step waiting on the GPU and 11% in explicit synchronizes, which is what a GPU-bound worker looks like. Adding GPUs to the pool stopped paying off at two because the feeder cannot keep more of them busy.</div>
  <div class="finding binding"><b>Per 16 s window (fast, 10 steps, 2 GPUs):</b> 37 s waiting on workers, 15 s conditioning, 13 s decode and warp, 10 s restore, 3 s write. Four windows plus the fixed costs make the 274 s lipsync stage. The quality job does the same per window with 40 steps on three GPUs, so denoise dominates there (≈140 s per window) and the coordinator costs are proportionally smaller.</div>
  <div class="finding cold"><b>Cold start costs ~75 s on the first request after a renderer restart.</b> Pipeline build 24 s, worker pool 14 s, and 36 s creating the insightface ONNX sessions. The 36 s is almost entirely pure-Python protobuf parsing: the LatentSync image ships <code>protobuf 3.20.3</code> with the <code>python</code> implementation, and insightface loads all five buffalo_l models (340 MB) even though only the 17 MB detector and 5 MB landmark model are used. The batch coordinator capture caught this: 78% of its 60 s sample window was that initialization.</div>
  <div class="finding ok"><b>No hardware contention was observed.</b> Host CPU stayed under 2% of 344 threads, container CPU peaked at 12 cores against a 32-core quota, NVENC and NVDEC engines never exceeded 19%, and no GPU exceeded 36 GB of 96 GB. GPUs 4, 6, 7 reported PCIe gen 1 at idle, which is power management, not a fault. The fast pair (3, 5) spans both NUMA nodes, so the host staging of conditioning tensors crosses sockets; that is a cost of the design above, not a hardware limit.</div>
  <div class="finding ok"><b>Everything before lipsync is now small.</b> Transcribe 1 s (batched whisper, GPU 0), translate 1 s fast / 5 s quality (GPU 1 pinned at 100% while answering), TTS 12–18 s. TTS variance comes from same-text retakes: the first XTTS segment was synthesized three times (33.7 s, 25.4 s, 25.8 s of audio) before it fit its slot. XTTS runs GPU 0 at 43% and the backend at 2–3 CPU cores; 10% of its samples are CPU audio decoding.</div>
</section>

<section class="cols">
  <div>
    <h2>Fast job stages</h2>
    <div class="tablewrap"><table><thead><tr><th>Stage</th><th>Seconds</th><th>GPUs busy (mean, peak, VRAM)</th><th>Host CPU</th><th>Containers</th></tr></thead><tbody>{stage_rows('fast')}</tbody></table></div>
  </div>
  <div>
    <h2>Quality job stages</h2>
    <div class="tablewrap"><table><thead><tr><th>Stage</th><th>Seconds</th><th>GPUs busy (mean, peak, VRAM)</th><th>Host CPU</th><th>Containers</th></tr></thead><tbody>{stage_rows('quality')}</tbody></table></div>
  </div>
</section>

<section>
  <h2>Hot functions (py-spy, wall-clock sampling at 100 Hz)</h2>
  <p class="lede">Self time is where the thread was when sampled. For GPU work the sample lands on the first call that waits for the device, so a worker's "DDIM step" is mostly GPU time.</p>
  <div class="cols">
    <div><h3>Fast coordinator, one thread</h3><div class="tablewrap"><table><tbody>{hot_rows('fast-coordinator', 'self')}</tbody></table></div></div>
    <div><h3>Fast denoise worker (GPU-bound)</h3><div class="tablewrap"><table><tbody>{hot_rows('fast-worker', 'self')}</tbody></table></div></div>
    <div><h3>Batch coordinator, first request (cold)</h3><div class="tablewrap"><table><tbody>{hot_rows('batch-coordinator', 'self')}</tbody></table></div></div>
    <div><h3>Backend during TTS</h3><div class="tablewrap"><table><tbody>{hot_rows('backend-tts', 'self')}</tbody></table></div></div>
  </div>
</section>

<section>
  <h2>What to change, in order of payoff</h2>
  <ol>
    <li><b>Take conditioning off the coordinator.</b> Batch the VAE encode of masked frames for the whole window up front (it is per 16-frame chunk today), or move it into each worker so the worker encodes its own chunk. Either removes the 25% serial slice and lets the workers run back to back.</li>
    <li><b>Stop staging tensors through host memory.</b> torch.multiprocessing can share CUDA tensors by IPC handle; passing device tensors to the workers removes the 25% spent in <code>.to("cpu")</code> copies and the cross-NUMA hop. Keeping fast workers on one NUMA node (3 and 2, or 4 and 5) helps until then.</li>
    <li><b>Paste faces back on the GPU.</b> <code>restore_img</code> is cv2 warp-and-blend on 1080×1920 frames on one CPU thread. A grid_sample-based paste on the worker GPU, or at minimum a thread pool (cv2 releases the GIL), recovers the 20%.</li>
    <li><b>Fix the 36 s ONNX cold start.</b> Load insightface with <code>allowed_modules=["detection", "landmark_2d_106"]</code> so 318 MB of unused models are skipped, move the image to a protobuf build with the upb implementation, and build the ImageProcessor at service startup instead of on the first request.</li>
    <li><b>Then add GPU replicas.</b> With the coordinator fixed, VRAM headroom (35 of 96 GB in use) allows two workers per GPU; until then more replicas only lower per-GPU utilization further.</li>
    <li><b>Cap TTS retakes by expected gain.</b> Three full re-synthesis passes of a 25 s segment cost 15 s. Rendering candidate takes in parallel or stopping when the second take is within 2% of the first would make TTS a flat 10 s.</li>
  </ol>
</section>

<section>
  <h2>Hardware and layout facts behind the numbers</h2>
  <div class="tablewrap"><table><thead><tr><th>GPU</th><th>Role in this capture</th><th>Peak VRAM</th><th>NUMA node</th></tr></thead><tbody>
  {''.join(f"<tr><td>{i}</td><td>{GPU_ROLE[i]}</td><td class=num>{max(v for v in T['gpu_mem_gb'][i] if v is not None):.0f} GB</td><td>{0 if int(i) < 4 else 1}</td></tr>" for i in sorted(T['gpu_mem_gb'], key=int))}
  </tbody></table></div>
  <p class="lede">Host: 2× Xeon 6787P, 172 cores / 344 threads, 2 TB RAM, GPUs 0–3 on socket 0 and 4–7 on socket 1, PCIe between sockets (no NVLink). Each renderer container has a 32-CPU quota it never approached.</p>
</section>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const D = {json.dumps(data)};
const css = getComputedStyle(document.documentElement);
const palette = ["#1f5f8b","#7a4fa0","#2e8b8b","#1f77b4","#d66027","#5fa3d9","#b7791f","#e07b39"];
const fg = () => css.getPropertyValue("--fg").trim(), muted = () => css.getPropertyValue("--muted").trim(), rule = () => css.getPropertyValue("--rule").trim();
const bandPlugin = {{ id: "bands", beforeDraw(chart) {{
  const {{ctx, chartArea, scales}} = chart; if (!chartArea) return;
  ctx.save();
  for (const b of D.bands) {{
    const x0 = scales.x.getPixelForValue(b.start), x1 = scales.x.getPixelForValue(b.end);
    ctx.fillStyle = b.color; ctx.fillRect(x0, chartArea.top, Math.max(1, x1 - x0), chartArea.bottom - chartArea.top);
    if (x1 - x0 > 28 && chart.canvas.id === "util") {{ ctx.fillStyle = muted(); ctx.font = "11px IBM Plex Sans, sans-serif"; ctx.fillText(b.stage, x0 + 3, chartArea.top + 12); }}
  }}
  ctx.restore();
}} }};
function series(src, label) {{
  return Object.keys(src).sort((a,b)=>a-b).map((i, k) => ({{ label: `GPU ${{i}}`, data: D.t.map((t, j) => ({{x: t, y: src[i][j]}})), borderColor: palette[k], borderWidth: 1.2, pointRadius: 0, tension: 0.15, spanGaps: true }}));
}}
const common = (ymax, ylabel) => ({{ animation: false, responsive: true, parsing: false, normalized: true,
  plugins: {{ legend: {{ display: false }}, tooltip: {{ mode: "nearest", intersect: false }} }},
  scales: {{ x: {{ type: "linear", title: {{ display: true, text: "seconds since capture start", color: muted() }}, ticks: {{ color: muted() }}, grid: {{ color: rule() }} }},
             y: {{ min: 0, max: ymax, title: {{ display: true, text: ylabel, color: muted() }}, ticks: {{ color: muted() }}, grid: {{ color: rule() }} }} }} }});
const utilDs = series(D.gpu);
utilDs.push({{ label: "host CPU %", data: D.host_t.map((t, j) => ({{x: t, y: D.host_cpu[j]}})), borderColor: fg(), borderDash: [4, 3], borderWidth: 1, pointRadius: 0, spanGaps: true }});
new Chart(document.getElementById("util"), {{ type: "line", data: {{ datasets: utilDs }}, options: common(100, "utilization %"), plugins: [bandPlugin] }});
new Chart(document.getElementById("mem"), {{ type: "line", data: {{ datasets: series(D.mem) }}, options: common(100, "VRAM used, GB"), plugins: [bandPlugin] }});
document.getElementById("legend").innerHTML = utilDs.map(d => `<span><i style="background:${{d.borderColor}}"></i>${{d.label}}</span>`).join("");
</script>
"""
OUT.write_text(page)
print("wrote", OUT, len(page) // 1024, "KB")
