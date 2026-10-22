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
    :root {{ color-scheme: dark; --ink:#f7f3e8; --muted:#aaa79e;
      --panel:#171916; --panel2:#20231e; --line:#353931; --accent:#f3c34f;
      --accent2:#6ed3ae; --danger:#ed7b74; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; background:radial-gradient(circle at 50% 0,#20231c 0,#0d0e0c 38%); color:var(--ink);
      font:15px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1320px,calc(100% - 32px)); margin:0 auto; padding:36px 0 64px; }}
    header {{ display:flex; align-items:end; justify-content:space-between; gap:24px;
      margin-bottom:24px; }}
    h1 {{ margin:0; font-size:clamp(27px,5vw,48px); line-height:1.05; letter-spacing:-.035em; }}
    .eyebrow {{ color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.13em;
      text-transform:uppercase; margin-bottom:8px; }}
    .model {{ color:var(--muted); text-align:right; }}
    .player {{ background:linear-gradient(145deg,#191b18,#131512); border:1px solid var(--line); border-radius:24px;
      padding:clamp(18px,3vw,32px); box-shadow:0 24px 80px #0008; }}
    audio {{ width:100%; height:42px; }}
    .now {{ display:grid; grid-template-columns:1fr minmax(240px,2fr) 1fr; align-items:center;
      gap:18px; min-height:175px; margin:14px 0 20px; text-align:center; }}
    .side {{ color:var(--muted); }} .side strong {{ display:block; color:#d8d5cc;
      font-size:clamp(20px,3vw,30px); margin-top:5px; }}
    .current {{ font-size:clamp(64px,13vw,132px); font-weight:850; line-height:.95;
      letter-spacing:-.07em; color:var(--accent); text-shadow:0 7px 30px #e9b94922; }}
    .time {{ color:var(--muted); font-variant-numeric:tabular-nums; margin-top:14px; }}
    .truth-now {{ display:inline-flex; align-items:center; gap:8px; margin-top:12px; padding:6px 11px;
      border:1px solid var(--line); border-radius:999px; color:var(--muted); background:#0e100d; }}
    .truth-now strong {{ color:var(--ink); font-size:17px; }}
    .agreement {{ width:9px; height:9px; border-radius:50%; background:var(--muted); }}
    .agreement.match {{ background:var(--accent2); box-shadow:0 0 12px var(--accent2); }}
    .agreement.mismatch {{ background:var(--danger); }}
    .rolling-head {{ display:flex; justify-content:space-between; align-items:center;
      color:var(--muted); margin:6px 2px 9px; font-size:13px; }}
    .rolling-head strong {{ color:var(--ink); font-size:15px; }}
    .rolling {{ display:grid; grid-template-columns:repeat(6,minmax(142px,1fr)); gap:10px;
      overflow-x:auto; padding:5px 3px 16px; scroll-snap-type:x proximity; }}
    .roll-card {{ position:relative; min-height:132px; border:1px solid var(--line); border-radius:16px;
      background:#10120f; color:var(--ink); padding:14px; text-align:left; cursor:pointer;
      overflow:hidden; scroll-snap-align:center; transition:transform .2s,border-color .2s,opacity .2s; }}
    .roll-card::before {{ content:''; position:absolute; inset:0 auto 0 0; width:5px; background:var(--card-color); }}
    .roll-card:hover {{ border-color:#777d70; transform:translateY(-2px); }}
    .roll-card.past {{ opacity:.48; }}
    .roll-card.now-card {{ border-color:var(--accent); background:#292515;
      box-shadow:0 0 0 1px var(--accent),0 10px 35px #0006; transform:translateY(-3px); }}
    .roll-role {{ color:var(--muted); font-size:10px; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }}
    .roll-chord {{ display:block; margin:7px 0 5px; font-size:clamp(25px,3vw,38px); font-weight:850; line-height:1; }}
    .roll-time,.roll-truth {{ display:block; color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
    .roll-truth {{ margin-top:8px; padding-top:7px; border-top:1px solid var(--line); }}
    .roll-truth strong {{ color:var(--ink); font-size:13px; }}
    @media (max-width:850px) {{ .rolling {{ grid-template-columns:repeat(6,155px); }} }}
    @media (max-width:650px) {{ header {{ align-items:start; flex-direction:column; }}
      .model {{ text-align:left; }} .now {{ grid-template-columns:1fr 2fr 1fr; min-height:150px; }}
      .current {{ font-size:58px; }} .side {{ font-size:11px; }} .side strong {{ font-size:17px; }}
      .roll-card {{ min-height:120px; }} }}
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
      <div><div class="current" id="current">—</div><div class="time" id="time">0:00 / 0:00</div>
        {('<div class="truth-now">Ground truth <strong id="ground-truth">—</strong><span class="agreement" id="agreement"></span></div>') if reference_events else ''}</div>
      <div class="side">Next<strong id="next">—</strong></div>
    </div>
    <div class="rolling-head"><strong>Chord sequence</strong><span>Previous · now · upcoming</span></div>
    <div class="rolling" id="rolling"></div>
  </section>
</main>
<script>
const events = {event_json};
const referenceEvents = {reference_json};
const audio = document.getElementById('audio');
const current = document.getElementById('current');
const previous = document.getElementById('previous');
const next = document.getElementById('next');
const clock = document.getElementById('time');
const groundTruth = document.getElementById('ground-truth');
const agreement = document.getElementById('agreement');
const rolling = document.getElementById('rolling');
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
const truthAt = time => referenceEvents.find(event => time >= event.start && time < event.end);
function renderRolling(activeIndex) {{
  rolling.innerHTML='';
  const first=Math.max(0,Math.min(activeIndex-1,Math.max(0,events.length-6)));
  events.slice(first,first+6).forEach((event,localIndex) => {{
    const index=first+localIndex; const offset=index-activeIndex;
    const truth=truthAt((event.start+event.end)/2);
    const role=offset===0?'Now':offset<0?'Previous':offset===1?'Next':`Coming +${{offset}}`;
    const card=document.createElement('button'); card.type='button';
    card.className=`roll-card ${{offset===0?'now-card':offset<0?'past':'future'}}`;
    card.style.setProperty('--card-color',palette[rootIndex(event.chord)]);
    card.innerHTML=`<span class="roll-role">${{role}}</span><strong class="roll-chord">${{event.chord}}</strong>
      <span class="roll-time">${{precise(event.start)}}–${{precise(event.end)}}</span>
      ${{truth?`<span class="roll-truth">Ground truth <strong>${{truth.chord}}</strong></span>`:''}}`;
    card.addEventListener('click',()=>{{audio.currentTime=event.start;audio.play();}});
    rolling.appendChild(card);
  }});
}}
let active=-1;
function update() {{
  const time=audio.currentTime || 0;
  let index=events.findIndex(event => time >= event.start && time < event.end);
  if (index < 0) index = time >= duration ? events.length-1 : 0;
  current.textContent=events[index].chord;
  if (groundTruth) {{
    const truth = truthAt(time);
    groundTruth.textContent = truth ? truth.chord : '—';
    agreement.className=`agreement ${{truth ? (truth.chord===events[index].chord?'match':'mismatch') : ''}}`;
    agreement.title=truth ? (truth.chord===events[index].chord?'Prediction matches':'Prediction differs') : '';
  }}
  previous.textContent=index ? events[index-1].chord : '—';
  next.textContent=index+1<events.length ? events[index+1].chord : '—';
  clock.textContent=`${{format(time)}} / ${{format(audio.duration || duration)}}`;
  if (index !== active) {{
    active=index;
    renderRolling(index);
  }}
}}
audio.addEventListener('timeupdate',update); audio.addEventListener('loadedmetadata',update);
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
