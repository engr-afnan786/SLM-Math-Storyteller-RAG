"""Tests for coherence tracking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import CoherenceTracker, EntityTracker


class MockEmbed:
    def embed_query(self, text):
        np.random.seed(hash(text) % 2**31)
        v = np.random.randn(384)
        return v / np.linalg.norm(v)


def test_single_turn():
    t = CoherenceTracker(MockEmbed())
    m = t.record_turn("hello", {"characters": [], "places": []})
    assert m["turn"] == 1
    assert m["topic_drift"] == 0.0

def test_identical_turns():
    t = CoherenceTracker(MockEmbed())
    t.record_turn("same text", {"characters": [], "places": []})
    m = t.record_turn("same text", {"characters": [], "places": []})
    assert m["topic_drift"] == 0.0

def test_summary():
    t = CoherenceTracker(MockEmbed())
    for i in range(5):
        t.record_turn(f"turn {i}", {"characters": ["Luna"], "places": []})
    s = t.get_summary()
    assert s["turns"] == 5

def test_entity_merge():
    a = {"characters": ["Luna"], "places": ["Cave"]}
    b = {"characters": ["Max"], "places": ["Cave"]}
    m = EntityTracker.merge(a, b)
    assert "Luna" in m["characters"]
    assert "Max" in m["characters"]

def test_reset():
    t = CoherenceTracker(MockEmbed())
    t.record_turn("hello", {"characters": [], "places": []})
    t.reset()
    assert len(t.turn_texts) == 0


if __name__ == "__main__":
    tests = [t for t in globals() if t.startswith("test_")]
    passed = failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed")