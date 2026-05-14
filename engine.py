"""
Core engine for the SLM Math Storyteller.
Handles: LLM loading, RAG retrieval, math verification,
entity tracking, coherence tracking, evaluation logging.
"""

import os
import re
import sys
import json
import time
import yaml
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    if not os.path.exists(config_path):
        print(f"[WARN] {config_path} not found, using defaults.")
        return _default_config()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _default_config() -> dict:
    return {
        "model": {"path": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                  "n_ctx": 2048, "n_batch": 512, "n_gpu_layers": 0,
                  "temperature": 0.7, "max_tokens": 200, "backend": "llama_cpp"},
        "openai": {"base_url": "http://127.0.0.1:1234/v1",
                   "api_key": "lm-studio", "model_name": "qwen2.5-1.5b-instruct"},
        "rag": {"index_path": "faiss_index", "knowledge_path": "Math_Project",
                "top_k": 3, "chunk_size": 500, "chunk_overlap": 50,
                "embedding_model": "all-MiniLM-L6-v2"},
        "coherence": {"drift_threshold": 0.4},
        "app": {"default_grade": 3, "share": False},
    }


# ── Math Verification (SymPy) ────────────────────────────────

def extract_arithmetic_facts(text: str) -> list[tuple]:
    """Find patterns like '3 + 4 = 7', '5 x 3 = 15', '1/2 = 2/4'."""
    facts = []
    for m in re.finditer(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', text):
        facts.append((m.group(1), '+', m.group(2), m.group(3)))
    for m in re.finditer(r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)', text):
        facts.append((m.group(1), '-', m.group(2), m.group(3)))
    for m in re.finditer(r'(\d+)\s*[x×\*]\s*(\d+)\s*=\s*(\d+)', text):
        facts.append((m.group(1), '*', m.group(2), m.group(3)))
    for m in re.finditer(r'(\d+)\s*[÷/]\s*(\d+)\s*=\s*(\d+)', text):
        # Avoid matching fractions like "1/2 of"
        facts.append((m.group(1), '/', m.group(2), m.group(3)))
    for m in re.finditer(r'(\d+)/(\d+)\s*=\s*(\d+)/(\d+)', text):
        facts.append((f"{m.group(1)}/{m.group(2)}", '==',
                       f"{m.group(3)}/{m.group(4)}", 'equal'))
    return facts


def verify_single_fact(a: str, op: str, b: str, result: str) -> tuple:
    """Verify one arithmetic fact. Returns (is_correct, corrected_result)."""
    try:
        from sympy import sympify, Rational, simplify
        if op == '==':
            left = simplify(Rational(a))
            right = simplify(Rational(b))
            return left == right, result
        expr_str = f"{a} {op} {b}"
        computed = sympify(expr_str)
        expected = float(result)
        computed_val = float(computed)
        is_correct = abs(computed_val - expected) < 1e-6
        corrected = str(int(computed)) if computed == int(computed) else f"{computed:.2f}"
        return is_correct, corrected
    except Exception:
        return False, result


def verify_math(text: str) -> tuple:
    """
    Check all arithmetic facts in text.
    Returns (corrected_text, all_correct, metrics_dict).
    """
    facts = extract_arithmetic_facts(text)
    metrics = {"total_facts": len(facts), "correct": 0, "incorrect": 0, "error_rate": 0.0}

    if not facts:
        return text, True, metrics

    corrected_text = text
    all_correct = True

    for a, op, b, result in facts:
        is_correct, correct_result = verify_single_fact(a, op, b, result)
        if is_correct:
            metrics["correct"] += 1
        else:
            metrics["incorrect"] += 1
            all_correct = False
            if op == '==':
                old = f"{a} = {b}"
                new = f"{a} = {b}"
            else:
                op_display = {'*': 'x', '/': '÷'}.get(op, op)
                old = f"{a} {op_display} {b} = {result}"
                new = f"{a} {op_display} {b} = {correct_result}"
            corrected_text = corrected_text.replace(old, new, 1)

    metrics["error_rate"] = (
        round(metrics["incorrect"] / metrics["total_facts"], 4)
        if metrics["total_facts"] > 0 else 0.0
    )
    return corrected_text, all_correct, metrics


# ── Entity Tracking (spaCy) ──────────────────────────────────

class EntityTracker:
    """Track story characters and places across turns."""

    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("[ENTITY] spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract(self, text: str) -> dict:
        if not self.nlp:
            return {"characters": [], "places": []}
        doc = self.nlp(text)
        characters = set()
        places = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                characters.add(ent.text)
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                places.add(ent.text)
        return {"characters": list(characters), "places": list(places)}

    @staticmethod
    def merge(existing: dict, new: dict) -> dict:
        return {
            "characters": list(set(existing.get("characters", []) + new.get("characters", []))),
            "places": list(set(existing.get("places", []) + new.get("places", []))),
        }

    @staticmethod
    def consistency_score(all_entities: list, window: int = 5) -> float:
        if len(all_entities) <= 1:
            return 1.0
        scores = []
        for i in range(1, len(all_entities)):
            known_chars = set()
            known_places = set()
            for j in range(i):
                known_chars.update(all_entities[j].get("characters", []))
                known_places.update(all_entities[j].get("places", []))
            total = len(known_chars) + len(known_places)
            if total == 0:
                scores.append(1.0)
                continue
            recent_chars = set()
            recent_places = set()
            start = max(0, i - window)
            for j in range(start, i + 1):
                recent_chars.update(all_entities[j].get("characters", []))
                recent_places.update(all_entities[j].get("places", []))
            retained = len(recent_chars & known_chars) + len(recent_places & known_places)
            scores.append(retained / total)
        return round(sum(scores) / len(scores), 4) if scores else 1.0


# ── Coherence Tracker ────────────────────────────────────────

class CoherenceTracker:
    """Track narrative coherence via topic drift and entity consistency."""

    def __init__(self, embed_model, drift_threshold: float = 0.4):
        self.embed_model = embed_model
        self.drift_threshold = drift_threshold
        self.turn_embeddings = []
        self.turn_texts = []
        self.drift_scores = []
        self.entities_per_turn = []

    def record_turn(self, text: str, entities: dict) -> dict:
        emb = np.array(self.embed_model.embed_query(text))
        self.turn_embeddings.append(emb)
        self.turn_texts.append(text)
        self.entities_per_turn.append(entities)

        metrics = {"turn": len(self.turn_texts), "topic_drift": 0.0,
                   "cumulative_drift": 0.0, "drift_detected": False}

        if len(self.turn_embeddings) > 1:
            prev = self.turn_embeddings[-2]
            curr = self.turn_embeddings[-1]
            norm_prod = np.linalg.norm(prev) * np.linalg.norm(curr)
            sim = float(np.dot(prev, curr) / norm_prod) if norm_prod > 0 else 1.0
            drift = 1.0 - sim
            self.drift_scores.append(drift)
            metrics["topic_drift"] = round(drift, 4)
            metrics["cumulative_drift"] = round(float(np.mean(self.drift_scores)), 4)
            metrics["drift_detected"] = drift > self.drift_threshold

        return metrics

    def get_summary(self) -> dict:
        n = len(self.turn_texts)
        if n == 0:
            return {"turns": 0, "avg_topic_drift": 0, "max_topic_drift": 0,
                    "drift_trend": 0, "entity_consistency": 1.0,
                    "drift_detected_count": 0}

        entity_cons = EntityTracker.consistency_score(self.entities_per_turn)
        avg_drift = float(np.mean(self.drift_scores)) if self.drift_scores else 0.0
        max_drift = float(np.max(self.drift_scores)) if self.drift_scores else 0.0

        drift_trend = 0.0
        if len(self.drift_scores) >= 4:
            x = np.arange(len(self.drift_scores))
            drift_trend = float(np.polyfit(x, np.array(self.drift_scores), 1)[0])

        return {
            "turns": n,
            "avg_topic_drift": round(avg_drift, 4),
            "max_topic_drift": round(max_drift, 4),
            "drift_trend": round(drift_trend, 4),
            "entity_consistency": round(entity_cons, 4),
            "drift_detected_count": sum(1 for d in self.drift_scores if d > self.drift_threshold),
        }

    def reset(self):
        self.turn_embeddings.clear()
        self.turn_texts.clear()
        self.drift_scores.clear()
        self.entities_per_turn.clear()


# ── Main Engine ───────────────────────────────────────────────

class StoryEngine:
    """Unified engine: LLM + RAG + math verification + coherence tracking."""

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.script_dir = os.path.dirname(os.path.abspath(config_path)) or "."
        self.llm = None
        self.openai_client = None
        self.vector_db = None
        self.embed_model = None
        self.entity_tracker = EntityTracker()
        self.coherence_tracker = None
        self.all_entities = []
        self.math_metrics_log = []
        self.session_history = []  # list of {"role": ..., "content": ...}
        self._initialized = False
        self._init_error = None

    def initialize(self) -> str:
        """Initialize all components. Returns status message."""
        try:
            self._load_llm()
            self._load_rag()
            self._initialized = True
            return "✅ System ready"
        except Exception as e:
            self._init_error = str(e)
            return f"❌ Error: {e}"

    def _load_llm(self):
        backend = self.cfg["model"].get("backend", "llama_cpp")
        if backend == "llama_cpp":
            from llama_cpp import Llama
            model_path = self.cfg["model"]["path"]
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.script_dir, model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Download a GGUF model and place it there, or edit config.yaml."
                )
            print(f"[LLM] Loading {model_path} ...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.cfg["model"].get("n_ctx", 2048),
                n_batch=self.cfg["model"].get("n_batch", 512),
                n_gpu_layers=self.cfg["model"].get("n_gpu_layers", 0),
                verbose=False,
            )
            print("[LLM] Model loaded.")
        else:
            from openai import OpenAI
            self.openai_client = OpenAI(
                base_url=self.cfg["openai"]["base_url"],
                api_key=self.cfg["openai"]["api_key"],
            )
            print("[LLM] OpenAI backend configured.")

    def _load_rag(self):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        emb_name = self.cfg["rag"]["embedding_model"]
        print(f"[RAG] Loading embedding model '{emb_name}' ...")
        self.embed_model = HuggingFaceEmbeddings(
            model_name=emb_name,
            model_kwargs={"local_files_only": False},
        )

        index_dir = os.path.join(self.script_dir, self.cfg["rag"]["index_path"])
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            print("[RAG] Loading existing FAISS index ...")
            self.vector_db = FAISS.load_local(
                index_dir, self.embed_model, allow_dangerous_deserialization=True
            )
            print(f"[RAG] Loaded ({self.vector_db.index.ntotal} vectors).")
        else:
            print("[RAG] No index found. Building from knowledge base ...")
            self._build_index(index_dir)

        self.coherence_tracker = CoherenceTracker(
            self.embed_model,
            drift_threshold=self.cfg.get("coherence", {}).get("drift_threshold", 0.4),
        )

    def _build_index(self, index_dir: str):
        from langchain_community.vectorstores import FAISS
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import CharacterTextSplitter

        kb_path = os.path.join(self.script_dir, self.cfg["rag"]["knowledge_path"])
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Knowledge base folder not found: {kb_path}")

        documents = []
        for f in sorted(os.listdir(kb_path)):
            if f.endswith(".txt"):
                loader = TextLoader(os.path.join(kb_path, f))
                documents.extend(loader.load())

        if not documents:
            raise ValueError("No documents found in knowledge base.")

        splitter = CharacterTextSplitter(
            chunk_size=self.cfg["rag"].get("chunk_size", 500),
            chunk_overlap=self.cfg["rag"].get("chunk_overlap", 50),
        )
        docs = splitter.split_documents(documents)
        self.vector_db = FAISS.from_documents(docs, self.embed_model)

        os.makedirs(index_dir, exist_ok=True)
        self.vector_db.save_local(index_dir)
        print(f"[RAG] Built index with {self.vector_db.index.ntotal} vectors.")

    def _generate(self, messages: list[dict]) -> str:
        """Generate response from LLM."""
        cfg_m = self.cfg["model"]
        if self.llm:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=cfg_m.get("max_tokens", 200),
                temperature=cfg_m.get("temperature", 0.7),
            )
            return response["choices"][0]["message"]["content"].strip()
        else:
            resp = self.openai_client.chat.completions.create(
                model=self.cfg["openai"]["model_name"],
                messages=messages,
                max_tokens=cfg_m.get("max_tokens", 200),
                temperature=cfg_m.get("temperature", 0.7),
            )
            return resp.choices[0].message.content.strip()

    def _build_system_prompt(self, grade: int, use_rag: bool) -> str:
        return (
            f"You are a friendly, concise assistant.\n"
            f"RULES:\n"
            f"- Keep responses to 2-4 sentences maximum.\n"
            f"- Use simple, everyday words anyone can understand.\n"
            f"- Be warm and human-friendly.\n"
            f"- Never use LaTeX. Write math in plain text (e.g. '1/2' not LaTeX).\n"
            f"\n"
            f"When telling MATH STORIES (Grade {grade}):\n"
            f"- Include ONE simple math problem fitting grade {grade}.\n"
            f"- Make it fun — like an adventure the reader is part of.\n"
            f"- End with a short question or choice to continue.\n"
            f"- Keep using the same characters and places already mentioned.\n"
            f"- If reference material is provided, use it for math facts and name the source file.\n"
            f"- If reference material is NOT relevant to the question, ignore it.\n"
            f"\n"
            f"When answering GENERAL questions:\n"
            f"- Answer directly and clearly.\n"
            f"- Be brief and helpful.\n"
            f"{'- If no reference material is given, say you do not have that info in your files.' if not use_rag else ''}"
        )

    def chat(self, message: str, use_rag: bool = True, grade: int = 3) -> dict:
        """
        Process one turn. Returns dict with:
          response, math_ok, math_metrics, coherence_metrics,
          entities, latency, context_used
        """
        if not self._initialized:
            return {"response": f"System not ready: {self._init_error}", "error": True}

        start = time.time()

        # ── Retrieve context ──
        context_used = ""
        if use_rag and self.vector_db:
            top_k = self.cfg["rag"].get("top_k", 3)
            results = self.vector_db.similarity_search(message, k=top_k)
            if results:
                parts = []
                for d in results:
                    src = os.path.basename(d.metadata.get("source", "Unknown"))
                    parts.append(f"[Source: {src}]\n{d.page_content}")
                context_used = "\n---\n".join(parts)

        # ── Build messages ──
        system_prompt = self._build_system_prompt(grade, use_rag)
        messages = [{"role": "system", "content": system_prompt}]

        # Add context as system note
        if context_used:
            messages.append({"role": "system", "content": f"Reference material:\n{context_used}"})

        # Add chat history (last 6 exchanges to stay within context)
        messages.extend(self.session_history[-12:])

        # Add user message
        messages.append({"role": "user", "content": message})

        # ── Generate ──
        raw_response = self._generate(messages)
        latency = time.time() - start

        # ── Math verification ──
        corrected, math_ok, math_metrics = verify_math(raw_response)

        # ── Entity tracking ──
        new_ents = self.entity_tracker.extract(message + " " + corrected)
        current_entities = (
            EntityTracker.merge(self.all_entities[-1], new_ents)
            if self.all_entities else new_ents
        )
        self.all_entities.append(current_entities)

        # ── Coherence tracking ──
        coh_metrics = {}
        if self.coherence_tracker:
            coh_metrics = self.coherence_tracker.record_turn(corrected, current_entities)

        self.math_metrics_log.append(math_metrics)

        # ── Update history ──
        self.session_history.append({"role": "user", "content": message})
        self.session_history.append({"role": "assistant", "content": corrected})

        # ── Log to file ──
        self._log_turn(message, corrected, context_used, latency,
                        math_metrics, coh_metrics, current_entities, use_rag, grade)

        return {
            "response": corrected,
            "math_ok": math_ok,
            "math_metrics": math_metrics,
            "coherence_metrics": coh_metrics,
            "entities": current_entities,
            "latency": round(latency, 2),
            "context_used": context_used,
            "error": False,
        }

    def get_metrics(self) -> dict:
        """Get current session metrics."""
        total_facts = sum(m.get("total_facts", 0) for m in self.math_metrics_log)
        total_correct = sum(m.get("correct", 0) for m in self.math_metrics_log)
        total_incorrect = sum(m.get("incorrect", 0) for m in self.math_metrics_log)
        error_rate = round(total_incorrect / total_facts, 4) if total_facts > 0 else 0.0

        coh = self.coherence_tracker.get_summary() if self.coherence_tracker else {}

        return {
            "turns": len(self.session_history) // 2,
            "math_total_facts": total_facts,
            "math_correct": total_correct,
            "math_incorrect": total_incorrect,
            "math_error_rate": error_rate,
            "coherence": coh,
            "entities": self.all_entities[-1] if self.all_entities else {},
        }

    def reset_session(self):
        """Reset for a new session."""
        self.session_history.clear()
        self.all_entities.clear()
        self.math_metrics_log.clear()
        self.entity_tracker = EntityTracker()
        if self.coherence_tracker:
            self.coherence_tracker.reset()

    def _log_turn(self, user, ai, context, latency, math_m, coh_m, entities, rag, grade):
        entry = {
            "turn": len(self.session_history) // 2,
            "rag_active": rag,
            "grade": grade,
            "user_input": user,
            "ai_response": ai,
            "context_retrieved": context[:500] if context else "",
            "latency_seconds": round(latency, 2),
            "math_metrics": math_m,
            "coherence_metrics": coh_m,
            "entities": entities,
        }
        fname = "With_RAG.json" if rag else "Without_RAG.json"
        path = os.path.join(self.script_dir, fname)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")