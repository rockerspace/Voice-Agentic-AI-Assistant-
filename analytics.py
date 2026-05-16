"""
Conversation Analytics Dashboard
Reads conversations.csv and generates insights + HTML report
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CSV_FILE = "conversations.csv"
REPORT_FILE = "analytics_report.html"


def load_conversations(filepath: str) -> list:
    if not Path(filepath).exists():
        print(f"❌  CSV not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(rows: list) -> dict:
    if not rows:
        return {}

    user_rows = [r for r in rows if r["speaker"] == "user"]
    asst_rows = [r for r in rows if r["speaker"] == "assistant"]

    sessions = list({r["session_id"] for r in rows})
    intents = Counter(r["intent"] for r in user_rows if r["intent"])
    sentiments = Counter(r["sentiment"] for r in user_rows if r["sentiment"])

    proc_times = []
    for r in asst_rows:
        try:
            proc_times.append(float(r["processing_time_ms"]))
        except (ValueError, KeyError):
            pass

    avg_proc = sum(proc_times) / len(proc_times) if proc_times else 0

    # Per-session turn counts
    session_turns = defaultdict(int)
    for r in user_rows:
        session_turns[r["session_id"]] += 1

    # Timeline: turns per date
    timeline = defaultdict(int)
    for r in user_rows:
        try:
            date = r["timestamp"][:10]
            timeline[date] += 1
        except Exception:
            pass

    return {
        "total_records": len(rows),
        "total_sessions": len(sessions),
        "total_user_turns": len(user_rows),
        "total_asst_turns": len(asst_rows),
        "top_intents": intents.most_common(10),
        "sentiment_dist": dict(sentiments),
        "avg_processing_ms": round(avg_proc, 1),
        "min_processing_ms": round(min(proc_times), 1) if proc_times else 0,
        "max_processing_ms": round(max(proc_times), 1) if proc_times else 0,
        "session_turns": dict(session_turns),
        "timeline": dict(sorted(timeline.items())),
        "stt_engines": Counter(r["stt_engine"] for r in user_rows if r["stt_engine"]).most_common(),
        "tts_engines": Counter(r["tts_engine"] for r in asst_rows if r["tts_engine"]).most_common(),
    }


def generate_html_report(stats: dict, rows: list) -> str:
    recent = rows[-20:][::-1]  # last 20 rows, newest first

    intent_labels = json.dumps([i[0] for i in stats.get("top_intents", [])])
    intent_values = json.dumps([i[1] for i in stats.get("top_intents", [])])

    sentiment = stats.get("sentiment_dist", {})
    sent_labels = json.dumps(list(sentiment.keys()))
    sent_values = json.dumps(list(sentiment.values()))

    timeline = stats.get("timeline", {})
    tl_labels = json.dumps(list(timeline.keys()))
    tl_values = json.dumps(list(timeline.values()))

    recent_html = ""
    for r in recent:
        badge_color = {"user": "#4f46e5", "assistant": "#059669"}.get(r["speaker"], "#6b7280")
        recent_html += f"""
        <tr>
          <td>{r.get('timestamp','')[:19]}</td>
          <td><span style="background:{badge_color};color:#fff;padding:2px 8px;border-radius:12px;font-size:12px">{r.get('speaker','')}</span></td>
          <td>{r.get('raw_text') or r.get('response_text','')[:80]}</td>
          <td>{r.get('intent','')}</td>
          <td>{r.get('sentiment','')}</td>
          <td>{r.get('processing_time_ms','') or '-'}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice AI Assistant — Analytics Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
             padding: 32px 40px; display: flex; align-items: center; gap: 16px; }}
  .header h1 {{ font-size: 28px; font-weight: 700; color: #fff; }}
  .header p {{ color: rgba(255,255,255,0.8); font-size: 14px; margin-top: 4px; }}
  .icon {{ font-size: 40px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 20px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px;
           text-align: center; transition: transform .2s; }}
  .kpi:hover {{ transform: translateY(-3px); }}
  .kpi .val {{ font-size: 36px; font-weight: 800; color: #818cf8; }}
  .kpi .lbl {{ font-size: 13px; color: #94a3b8; margin-top: 6px; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 24px; margin-bottom: 32px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; }}
  .card h3 {{ font-size: 16px; font-weight: 600; color: #c7d2fe; margin-bottom: 20px;
              border-bottom: 1px solid #334155; padding-bottom: 10px; }}
  .chart-wrap {{ position: relative; height: 260px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead tr {{ background: #1e293b; }}
  th {{ padding: 10px 12px; text-align: left; color: #94a3b8; font-weight: 600;
        border-bottom: 2px solid #334155; white-space: nowrap; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e293b; color: #cbd5e1; vertical-align: middle; }}
  tbody tr:hover {{ background: #1e293b; }}
  tbody tr:nth-child(even) {{ background: #0f172a; }}
  .perf {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .perf-item {{ background: #0f172a; border-radius: 8px; padding: 14px 20px; flex: 1; min-width: 100px; text-align: center; }}
  .perf-item .num {{ font-size: 24px; font-weight: 700; color: #34d399; }}
  .perf-item .lbl {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
  .footer {{ text-align: center; padding: 24px; color: #475569; font-size: 12px; }}
  .badge-grid {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }}
  .badge {{ background: #0f172a; border-radius: 20px; padding: 6px 14px; font-size: 13px; }}
</style>
</head>
<body>
<div class="header">
  <div class="icon">🤖</div>
  <div>
    <h1>Voice Agentic AI Assistant — Analytics Dashboard</h1>
    <p>Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} &nbsp;|&nbsp; Source: conversations.csv</p>
  </div>
</div>
<div class="container">

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi"><div class="val">{stats.get('total_records',0)}</div><div class="lbl">Total Records</div></div>
    <div class="kpi"><div class="val">{stats.get('total_sessions',0)}</div><div class="lbl">Sessions</div></div>
    <div class="kpi"><div class="val">{stats.get('total_user_turns',0)}</div><div class="lbl">User Turns</div></div>
    <div class="kpi"><div class="val">{stats.get('total_asst_turns',0)}</div><div class="lbl">AI Responses</div></div>
    <div class="kpi"><div class="val">{stats.get('avg_processing_ms',0)}ms</div><div class="lbl">Avg Response Time</div></div>
    <div class="kpi"><div class="val">{len(stats.get('top_intents',[]))}</div><div class="lbl">Unique Intents</div></div>
  </div>

  <!-- Charts Row -->
  <div class="charts-grid">
    <div class="card">
      <h3>📊 Top Intents Distribution</h3>
      <div class="chart-wrap"><canvas id="intentChart"></canvas></div>
    </div>
    <div class="card">
      <h3>😊 Sentiment Analysis</h3>
      <div class="chart-wrap"><canvas id="sentimentChart"></canvas></div>
    </div>
    <div class="card">
      <h3>📅 Daily Conversation Volume</h3>
      <div class="chart-wrap"><canvas id="timelineChart"></canvas></div>
    </div>
    <div class="card">
      <h3>⚡ Performance Metrics</h3>
      <div class="perf">
        <div class="perf-item"><div class="num">{stats.get('min_processing_ms',0)}ms</div><div class="lbl">Min Response</div></div>
        <div class="perf-item"><div class="num">{stats.get('avg_processing_ms',0)}ms</div><div class="lbl">Avg Response</div></div>
        <div class="perf-item"><div class="num">{stats.get('max_processing_ms',0)}ms</div><div class="lbl">Max Response</div></div>
      </div>
      <br>
      <h3 style="margin-top:12px">🎙️ STT Engines</h3>
      <div class="badge-grid">{''.join(f'<span class="badge">🎤 {e[0]}: {e[1]}</span>' for e in stats.get('stt_engines',[]))}</div>
      <br>
      <h3 style="margin-top:8px">🔊 TTS Engines</h3>
      <div class="badge-grid">{''.join(f'<span class="badge">🔊 {e[0]}: {e[1]}</span>' for e in stats.get('tts_engines',[]))}</div>
    </div>
  </div>

  <!-- Recent Conversations -->
  <div class="card">
    <h3>💬 Recent Conversations (Last 20 Records)</h3>
    <div style="overflow-x:auto">
      <table>
        <thead><tr><th>Timestamp</th><th>Speaker</th><th>Text</th><th>Intent</th><th>Sentiment</th><th>Time (ms)</th></tr></thead>
        <tbody>{recent_html}</tbody>
      </table>
    </div>
  </div>
</div>

<div class="footer">Voice Agentic AI Assistant &nbsp;|&nbsp; Powered by Python, RegEx NLP, Google Gemini / OpenAI &nbsp;|&nbsp; Analytics auto-generated from conversations.csv</div>

<script>
const C = (id, type, labels, data, opts={{}}) => {{
  new Chart(document.getElementById(id), {{
    type, data: {{ labels, datasets: [{{ data,
      backgroundColor: ['#818cf8','#34d399','#fb7185','#fbbf24','#60a5fa','#a78bfa','#f472b6','#4ade80','#38bdf8','#facc15'],
      borderColor: '#0f172a', borderWidth: 2, fill: type==='line', tension: 0.4, ...opts }}] }},
    options: {{ plugins: {{ legend: {{ display: type!=='bar', labels: {{ color: '#94a3b8', font: {{ size: 12 }} }} }} }},
               scales: type !== 'pie' && type !== 'doughnut' ? {{
                 x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45 }}, grid: {{ color: '#1e293b' }} }},
                 y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
               }} : {{}}
    }}
  }});
}};
C('intentChart',   'bar',      {intent_labels}, {intent_values});
C('sentimentChart','doughnut', {sent_labels},   {sent_values});
C('timelineChart', 'line',     {tl_labels},     {tl_values}, {{backgroundColor: 'rgba(129,140,248,0.15)', borderColor: '#818cf8', pointBackgroundColor: '#818cf8'}});
</script>
</body>
</html>"""


def main():
    print("📊  Voice AI Assistant — Analytics Generator")
    print("="*50)
    rows = load_conversations(CSV_FILE)
    if not rows:
        print("No data found. Run voice_assistant.py first!")
        return
    stats = compute_stats(rows)
    print(f"  Total records    : {stats['total_records']}")
    print(f"  Total sessions   : {stats['total_sessions']}")
    print(f"  User turns       : {stats['total_user_turns']}")
    print(f"  Avg response time: {stats['avg_processing_ms']} ms")
    print(f"\n  Top intents:")
    for intent, count in stats.get("top_intents", [])[:5]:
        print(f"    {intent:<25} {count}")
    html = generate_html_report(stats, rows)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅  HTML report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()