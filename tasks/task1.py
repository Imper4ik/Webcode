"""Набор тестов для проверки реализации ``my_len``."""

from typing import Callable


def run_tests(my_len: Callable[[str], int]) -> None:
    """Проверяет корректность пользовательской реализации ``my_len``."""

    assert my_len("abc") == 3
    assert my_len("") == 0
    assert my_len("привет") == 6
    print("Task1 OK")


__all__ = ["run_tests"]
