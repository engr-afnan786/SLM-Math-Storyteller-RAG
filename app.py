#!/usr/bin/env python3
"""
Professional Gradio UI for SLM Math Storyteller with RAG.
Run: python app.py
"""

import os
import json
import gradio as gr
from engine import StoryEngine

# ── Constants ──

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STORY_STARTERS = [
    ("🐉 Dragon Quest", "Tell me a story about a dragon who counts treasure"),
    ("🏴‍☠️ Pirate Fractions", "A pirate finds a map with fraction puzzles"),
    ("🚀 Space Math", "An astronaut learns multiplication in space"),
    ("🧚 Fairy Geometry", "A fairy builds a castle using geometry"),
    ("🤖 Robot Ratios", "A robot discovers ratios while cooking"),
    ("🎲 Math Riddle", "Give me a fun math riddle to solve"),
]

CSS = """
#main-chat { min-height: 450px; }
.sidebar-section { padding: 8px 0; }
.metric-row { display: flex; justify-content: space-between; padding: 2px 0; }
.metric-label { color: #64748b; font-size: 0.85em; }
.metric-value { color: #1e293b; font-weight: 600; font-size: 0.9em; }
.status-bar { padding: 6px 12px; border-radius: 8px; font-size: 0.85em; margin-top: 4px; }
.story-btn { min-height: 38px !important; font-size: 0.9em !important; }
"""

# ── Engine ──

engine = StoryEngine(os.path.join(SCRIPT_DIR, "config.yaml"))
init_status = engine.initialize()


# ── Handler Functions ──

def respond(message, chat_history, use_rag, grade):
    """Process one chat turn."""
    if not message.strip():
        return "", chat_history, _metrics_text(), _status_html("", use_rag)

    result = engine.chat(message, use_rag=use_rag, grade=int(grade))

    if result.get("error"):
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": result["response"]})
        return "", chat_history, _metrics_text(), _status_html("error", use_rag)

    # Build display response with verification badge
    response = result["response"]
    badge = "✅" if result["math_ok"] else "⚠️ Corrected"
    chars = ", ".join(result["entities"].get("characters", [])[:3])
    places = ", ".join(result["entities"].get("places", [])[:3])
    meta_parts = [f"{badge} | {result['latency']}s"]
    if chars:
        meta_parts.append(f"Characters: {chars}")
    if places:
        meta_parts.append(f"Places: {places}")

    display = f"{response}\n\n_{' | '.join(meta_parts)}_"
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": display})

    status_type = "ok" if result["math_ok"] else "corrected"
    return "", chat_history, _metrics_text(), _status_html(status_type, use_rag)


def start_story(story_text, chat_history, use_rag, grade):
    """Fill the input with a story starter and auto-send."""
    return respond(story_text, chat_history, use_rag, grade)


def new_session():
    """Reset for a new session."""
    engine.reset_session()
    return [], "", _metrics_text(), _status_html("new", True)


def _metrics_text() -> str:
    m = engine.get_metrics()
    coh = m.get("coherence", {})
    turns = m.get("turns", 0)
    err = m.get("math_error_rate", 0)
    drift = coh.get("avg_topic_drift", 0)
    trend = coh.get("drift_trend", 0)
    ent_con = coh.get("entity_consistency", 1.0)
    alerts = coh.get("drift_detected_count", 0)

    return (
        f"**Turns:** {turns}  \n"
        f"**Math Error Rate:** {err}  \n"
        f"**Avg Topic Drift:** {drift}  \n"
        f"**Drift Trend:** {trend}  \n"
        f"**Entity Consistency:** {ent_con}  \n"
        f"**Drift Alerts:** {alerts}"
    )


def _status_html(status_type: str, use_rag: bool) -> str:
    rag_label = "RAG: ON" if use_rag else "RAG: OFF"
    rag_color = "#10b981" if use_rag else "#f59e0b"

    if status_type == "ok":
        msg, color = "✅ Math verified", "#10b981"
    elif status_type == "corrected":
        msg, color = "⚠️ Math corrected", "#f59e0b"
    elif status_type == "error":
        msg, color = "❌ Error", "#ef4444"
    elif status_type == "new":
        msg, color = "🔄 New session", "#3b82f6"
    else:
        msg, color = "● Ready", "#10b981"

    return (
        f'<div style="display:flex;gap:16px;align-items:center;padding:6px 12px;'
        f'background:#f8fafc;border-radius:8px;font-size:0.85em;">'
        f'<span style="color:{color};font-weight:600;">{msg}</span>'
        f'<span style="color:{rag_color};font-weight:600;">{rag_label}</span>'
        f'</div>'
    )


def save_human_score(flow, consistency, engagement, math_accuracy):
    """Save human evaluation scores."""
    scores_dir = os.path.join(SCRIPT_DIR, "evaluation_results")
    os.makedirs(scores_dir, exist_ok=True)
    path = os.path.join(scores_dir, "human_scores.json")
    scores = []
    if os.path.exists(path):
        with open(path, "r") as f:
            scores = json.load(f)
    scores.append({
        "turns": engine.get_metrics().get("turns", 0),
        "rag_active": True,
        "grade": engine.cfg.get("app", {}).get("default_grade", 3),
        "flow": int(flow),
        "consistency": int(consistency),
        "engagement": int(engagement),
        "math_accuracy": int(math_accuracy),
    })
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    return "✅ Score saved!"


# ── Build UI ──

app = gr.Blocks(title="SLM Math Storyteller")

with app:
    gr.Markdown(
        "# 🧮 SLM Math Storyteller — RAG Enhanced\n"
        "_Co-creative math adventures powered by a small language model with retrieval-augmented generation_"
    )

    with gr.Row():
        # ── Sidebar ──
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Settings")
            rag_toggle = gr.Checkbox(label="Knowledge Base (RAG)", value=True,
                                     info="Toggle RAG on/off for comparison")
            grade_select = gr.Dropdown(
                choices=["1", "2", "3", "4", "5", "6"],
                value=str(engine.cfg.get("app", {}).get("default_grade", 3)),
                label="Grade Level", info="Math difficulty level"
            )

            gr.Markdown("### 📖 Quick Start")
            starter_buttons = []
            for label, text in STORY_STARTERS:
                btn = gr.Button(label, variant="secondary", size="sm", elem_classes=["story-btn"])
                starter_buttons.append((btn, text))

            gr.Markdown("### 📊 Live Metrics")
            metrics_display = gr.Markdown(_metrics_text(), elem_id="metrics-display")

            new_session_btn = gr.Button("🔄 New Session", variant="secondary")

            gr.Markdown("### 📝 Rate This Session")
            with gr.Row():
                flow_s = gr.Slider(1, 5, 3, step=1, label="Flow")
                cons_s = gr.Slider(1, 5, 3, step=1, label="Consistency")
            with gr.Row():
                eng_s = gr.Slider(1, 5, 3, step=1, label="Engagement")
                math_s = gr.Slider(1, 5, 3, step=1, label="Math Accuracy")
            score_out = gr.Markdown("")
            gr.Button("Submit Score", variant="primary", size="sm").click(
                fn=save_human_score,
                inputs=[flow_s, cons_s, eng_s, math_s],
                outputs=[score_out],
            )

        # ── Main ──
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=480,
                elem_id="main-chat",
                avatar_images=(None, "🤖"),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message... (math story, question, or anything)",
                    scale=5, show_label=False, lines=1
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)
            status_bar = gr.HTML(_status_html("", True), elem_id="status-bar")

    # Wire main chat
    msg.submit(respond, [msg, chatbot, rag_toggle, grade_select],
               [msg, chatbot, metrics_display, status_bar])
    send_btn.click(respond, [msg, chatbot, rag_toggle, grade_select],
                   [msg, chatbot, metrics_display, status_bar])

    # Wire story starters
    for btn, text in starter_buttons:
        btn.click(
            fn=start_story,
            inputs=[gr.State(text), chatbot, rag_toggle, grade_select],
            outputs=[msg, chatbot, metrics_display, status_bar],
        )

    new_session_btn.click(
        fn=new_session,
        outputs=[chatbot, msg, metrics_display, status_bar],
    )

    gr.Markdown(f"<small>System: {init_status}</small>")

if __name__ == "__main__":
    share = engine.cfg.get("app", {}).get("share", False)
    app.launch(
        share=share,
        inbrowser=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=CSS,
    )
