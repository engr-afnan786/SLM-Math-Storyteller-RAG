"""Tests for math verification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import verify_math, extract_arithmetic_facts


def test_correct_addition():
    corrected, ok, m = verify_math("3 + 4 = 7")
    assert ok is True
    assert m["correct"] == 1
    assert m["error_rate"] == 0.0

def test_incorrect_addition():
    corrected, ok, m = verify_math("3 + 4 = 8")
    assert ok is False
    assert m["incorrect"] == 1
    assert "7" in corrected

def test_correct_multiplication():
    corrected, ok, m = verify_math("6 x 7 = 42")
    assert ok is True

def test_correct_subtraction():
    corrected, ok, m = verify_math("9 - 3 = 6")
    assert ok is True

def test_no_math_facts():
    corrected, ok, m = verify_math("The dragon flew away.")
    assert ok is True
    assert m["total_facts"] == 0

def test_multiple_facts():
    corrected, ok, m = verify_math("3 + 4 = 7 and 2 x 5 = 10")
    assert ok is True
    assert m["total_facts"] == 2

def test_mixed_correct_incorrect():
    corrected, ok, m = verify_math("3 + 4 = 7 but 2 + 2 = 5")
    assert ok is False
    assert m["error_rate"] == 0.5

def test_extract_patterns():
    facts = extract_arithmetic_facts("3 + 4 = 7, 10 - 3 = 7, 5 x 6 = 30")
    assert len(facts) == 3


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