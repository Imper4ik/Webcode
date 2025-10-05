"""Built-in training tasks."""
from __future__ import annotations

from typing import Dict


TASKS: Dict[str, Dict[str, str]] = {
    "task1": {
        "description": "Реализуй функцию my_len(s: str) -> int, которая возвращает длину строки без использования len().",
        "starter_code": """
# Напиши функцию ниже

def my_len(s: str) -> int:
    # TODO: посчитать длину s
    count = 0
    for _ in s:
        count += 1
    return count
""".strip(),
        "tests": """
# ---- ТЕСТЫ (не менять) ----
if __name__ == "__main__":
    assert my_len("") == 0
    assert my_len("abc") == 3
    assert my_len("привет") == 6
    print("OK")
""".strip(),
    },
    "task2": {
        "description": "Реализуй функцию square(x: int|float) -> int|float, которая возвращает x*x.",
        "starter_code": """
# Напиши функцию ниже

def square(x):
    return x * x
""".strip(),
        "tests": """
# ---- ТЕСТЫ (не менять) ----
if __name__ == "__main__":
    assert square(2) == 4
    assert square(-3) == 9
    assert square(1.5) == 2.25
    print("OK")
""".strip(),
    },
}


__all__ = ["TASKS"]
