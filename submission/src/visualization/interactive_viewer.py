"""Write a standalone audio player with synchronized chord annotations."""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import librosa
import soundfile as sf

from src.inference import ChordTimeline


def _relative_audio_url(audio_path: Path, viewer_path: Path) -> str:
    relative = os.path.relpath(audio_path.resolve(), viewer_path.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/._-~")


def _browser_audio(audio_path: Path, viewer_path: Path) -> Path:
    """Put a browser-playable copy beside the viewer.

    Keeping the media under the output directory also works when the viewer is
    served with ``python -m http.server`` from ``output/``.
    """

    media_dir = viewer_path.parent / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    if audio_path.suffix.lower() in {".mp3", ".wav", ".ogg"}:
        target = media_dir / audio_path.name
        if not target.is_file() or target.stat().st_mtime < audio_path.stat().st_mtime:
            shutil.copy2(audio_path, target)
        return target
    target = media_dir / f"{audio_path.stem}_browser.wav"
    if target.is_file() and target.stat().st_mtime >= audio_path.stat().st_mtime:
        return target
    audio, _ = librosa.load(str(audio_path), sr=22_050, mono=True)
    sf.write(target, audio, 22_050, subtype="PCM_16")
    return target


def write_interactive_viewer(
    song_name: str,
    audio_path: Path | str,
    timeline: ChordTimeline,
    output_path: Path | str,
    *,
    model_name: str,
    reference: ChordTimeline | None = None,
) -> Path:
    """Create one dependency-free HTML viewer and return its path."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "start": float(start),
            "end": float(end),
            "chord": str(label),
        }
        for start, end, label in zip(timeline.starts, timeline.ends, timeline.labels)
    ]
    if not events:
        raise ValueError("Cannot create an interactive viewer for an empty timeline.")
    event_json = json.dumps(events, ensure_ascii=False).replace("</", "<\\/")
    reference_events = []
    if reference is not None:
        reference_events = [
            {"start": float(start), "end": float(end), "chord": str(label)}
            for start, end, label in zip(
                reference.starts, reference.ends, reference.labels
            )
        ]
    reference_json = json.dumps(reference_events, ensure_ascii=False).replace("</", "<\\/")
    playable_audio = _browser_audio(Path(audio_path), target)
    audio_url = _relative_audio_url(playable_audio, target)
    title = html.escape(song_name)
    model = html.escape(model_name)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · Chord viewer</title>
  <style>
    :root {{ color-scheme: dark; --ink:#f5f2e9; --muted:#a8a49a;
      --panel:#181917; --line:#32342f; --accent:#e9b949; --accent2:#66c7a5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; background:#0d0e0c; color:var(--ink);
      font:15px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1120px,calc(100% - 32px)); margin:0 auto; padding:42px 0 64px; }}
    header {{ display:flex; align-items:end; justify-content:space-between; gap:24px;
      margin-bottom:24px; }}
    h1 {{ margin:0; font-size:clamp(27px,5vw,48px); line-height:1.05; letter-spacing:-.035em; }}
    .eyebrow {{ color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.13em;
      text-transform:uppercase; margin-bottom:8px; }}
    .model {{ color:var(--muted); text-align:right; }}
    .player {{ background:var(--panel); border:1px solid var(--line); border-radius:18px;
      padding:22px; box-shadow:0 20px 60px #0006; }}
    audio {{ width:100%; height:42px; }}
    .now {{ display:grid; grid-template-columns:1fr minmax(180px,2fr) 1fr; align-items:center;
      gap:18px; min-height:190px; margin:18px 0; text-align:center; }}
    .side {{ color:var(--muted); }} .side strong {{ display:block; color:#d8d5cc;
      font-size:clamp(20px,3vw,30px); margin-top:5px; }}
    .current {{ font-size:clamp(64px,13vw,132px); font-weight:850; line-height:.95;
      letter-spacing:-.07em; color:var(--accent); text-shadow:0 7px 30px #e9b94922; }}
    .time {{ color:var(--muted); font-variant-numeric:tabular-nums; margin-top:14px; }}
    .overview {{ position:relative; display:flex; height:72px; overflow:hidden;
      border:1px solid var(--line); border-radius:12px; background:#11120f; cursor:pointer; }}
    .segment {{ min-width:2px; height:100%; display:flex; align-items:center;
      justify-content:center; overflow:hidden; border-right:1px solid #0d0e0c99;
      font-size:11px; font-weight:700; color:#10110f; user-select:none; }}
    .segment span {{ overflow:hidden; text-overflow:ellipsis; padding:0 4px; }}
    .reference-label {{ color:var(--muted); margin:18px 2px 6px; font-size:13px; }}
    .reference-overview {{ position:relative; display:flex; height:42px; overflow:hidden;
      border:1px solid var(--line); border-radius:12px; background:#11120f; }}
    .playhead {{ position:absolute; z-index:4; top:0; bottom:0; width:3px;
      background:white; box-shadow:0 0 12px #fff; pointer-events:none; transform:translateX(-1px); }}
    .events-head {{ display:flex; justify-content:space-between; margin:24px 2px 10px; }}
    .events-head span:last-child {{ color:var(--muted); }}
    .event-list {{ display:flex; gap:8px; overflow-x:auto; scroll-behavior:smooth;
      padding:2px 2px 12px; scrollbar-color:#555 transparent; }}
    button.event {{ flex:0 0 116px; border:1px solid var(--line); border-radius:10px;
      background:#11120f; color:var(--ink); padding:13px 12px; text-align:left; cursor:pointer; }}
    button.event:hover {{ border-color:#777; }} button.event.active {{ border-color:var(--accent);
      background:#292515; box-shadow:inset 0 0 0 1px var(--accent); }}
    .event .chord {{ display:block; font-size:22px; font-weight:750; }}
    .event .stamp {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
    .hint {{ color:var(--muted); margin:18px 2px 0; font-size:13px; }}
    @media (max-width:650px) {{ header {{ align-items:start; flex-direction:column; }}
      .model {{ text-align:left; }} .now {{ grid-template-columns:1fr 2fr 1fr; min-height:150px; }}
      .current {{ font-size:58px; }} .side {{ font-size:11px; }} .side strong {{ font-size:17px; }} }}
  </style>
</head>
<body>
<main>
  <header><div><div class="eyebrow">Live chord transcription</div><h1>{title}</h1></div>
    <div class="model">Model<br><strong>{model}</strong></div></header>
  <section class="player">
    <audio id="audio" controls preload="metadata" src="{audio_url}"></audio>
    <div class="now">
      <div class="side">Previous<strong id="previous">—</strong></div>
      <div><div class="current" id="current">—</div><div class="time" id="time">0:00 / 0:00</div></div>
      <div class="side">Next<strong id="next">—</strong></div>
    </div>
    <div class="overview" id="overview"><div class="playhead" id="playhead"></div></div>
    {('<div class="reference-label">Ground truth: <strong id="ground-truth">—</strong></div><div class="reference-overview" id="reference-overview"></div>') if reference_events else ''}
    <div class="events-head"><strong>Chord events</strong><span id="event-count"></span></div>
    <div class="event-list" id="event-list"></div>
    <p class="hint">Click any chord to seek. Space plays or pauses; ← and → jump five seconds.</p>
  </section>
</main>
<script>
const events = {event_json};
const referenceEvents = {reference_json};
const audio = document.getElementById('audio');
const overview = document.getElementById('overview');
const list = document.getElementById('event-list');
const current = document.getElementById('current');
const previous = document.getElementById('previous');
const next = document.getElementById('next');
const clock = document.getElementById('time');
const playhead = document.getElementById('playhead');
const groundTruth = document.getElementById('ground-truth');
const duration = Math.max(...events.map(event => event.end));
const palette = ['#e9b949','#66c7a5','#e07a5f','#81b29a','#a98eda','#5fa8d3',
  '#f2a65a','#8ecae6','#c9ada7','#90be6d','#f28482','#b8b8ff'];
const rootIndex = label => {{
  const match = label.match(/^([A-G](?:b|#)?)/); if (!match) return 0;
  return {{C:0,'C#':1,Db:1,D:2,'D#':3,Eb:3,E:4,F:5,'F#':6,Gb:6,G:7,
    'G#':8,Ab:8,A:9,'A#':10,Bb:10,B:11}}[match[1]] ?? 0;
}};
const format = seconds => {{
  if (!Number.isFinite(seconds)) return '0:00';
  const minutes = Math.floor(seconds / 60); const rest = Math.floor(seconds % 60);
  return `${{minutes}}:${{String(rest).padStart(2,'0')}}`;
}};
const precise = seconds => {{
  const minutes = Math.floor(seconds / 60); const rest = (seconds % 60).toFixed(1);
  return `${{minutes}}:${{rest.padStart(4,'0')}}`;
}};
events.forEach((event, index) => {{
  const segment = document.createElement('div'); segment.className = 'segment';
  segment.style.width = `${{100 * (event.end-event.start) / duration}}%`;
  segment.style.background = palette[rootIndex(event.chord)];
  segment.title = `${{event.chord}} · ${{precise(event.start)}}–${{precise(event.end)}}`;
  segment.innerHTML = `<span>${{event.chord}}</span>`;
  segment.addEventListener('click', () => {{ audio.currentTime=event.start; audio.play(); }});
  overview.insertBefore(segment, playhead);
  const button = document.createElement('button'); button.className='event'; button.type='button';
  button.innerHTML=`<span class="chord">${{event.chord}}</span><span class="stamp">${{precise(event.start)}}–${{precise(event.end)}}</span>`;
  button.addEventListener('click', () => {{ audio.currentTime=event.start; audio.play(); }});
  button.dataset.index=index; list.appendChild(button);
}});
if (referenceEvents.length) {{
  const referenceOverview = document.getElementById('reference-overview');
  referenceEvents.forEach(event => {{
    const segment = document.createElement('div'); segment.className = 'segment';
    segment.style.width = `${{100 * (event.end-event.start) / duration}}%`;
    segment.style.background = palette[rootIndex(event.chord)];
    segment.title = `${{event.chord}} · ${{precise(event.start)}}–${{precise(event.end)}}`;
    segment.innerHTML = `<span>${{event.chord}}</span>`;
    segment.addEventListener('click', () => {{ audio.currentTime=event.start; audio.play(); }});
    referenceOverview.appendChild(segment);
  }});
}}
document.getElementById('event-count').textContent=`${{events.length}} events`;
let active=-1;
function update() {{
  const time=audio.currentTime || 0;
  let index=events.findIndex(event => time >= event.start && time < event.end);
  if (index < 0) index = time >= duration ? events.length-1 : 0;
  current.textContent=events[index].chord;
  if (groundTruth) {{
    const truth = referenceEvents.find(event => time >= event.start && time < event.end);
    groundTruth.textContent = truth ? truth.chord : '—';
  }}
  previous.textContent=index ? events[index-1].chord : '—';
  next.textContent=index+1<events.length ? events[index+1].chord : '—';
  clock.textContent=`${{format(time)}} / ${{format(audio.duration || duration)}}`;
  playhead.style.left=`${{Math.min(100,100*time/duration)}}%`;
  if (index !== active) {{
    if (active >= 0) list.children[active].classList.remove('active');
    active=index; const item=list.children[index]; item.classList.add('active');
    item.scrollIntoView({{behavior:'smooth',block:'nearest',inline:'center'}});
  }}
}}
audio.addEventListener('timeupdate',update); audio.addEventListener('loadedmetadata',update);
overview.addEventListener('click',event => {{
  const box=overview.getBoundingClientRect(); audio.currentTime=(event.clientX-box.left)/box.width*duration;
  audio.play();
}});
document.addEventListener('keydown', event => {{
  if (event.target.tagName === 'BUTTON') return;
  if (event.code === 'Space') {{ event.preventDefault(); audio.paused ? audio.play() : audio.pause(); }}
  if (event.code === 'ArrowLeft') audio.currentTime=Math.max(0,audio.currentTime-5);
  if (event.code === 'ArrowRight') audio.currentTime=Math.min(duration,audio.currentTime+5);
}});
update();
</script>
</body>
</html>
"""
    target.write_text(document, encoding="utf-8")
    return target
