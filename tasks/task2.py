"""Набор тестов для проверки реализации ``square``."""

from typing import Callable, Union

Number = Union[int, float]


def run_tests(square: Callable[[Number], Number]) -> None:
    """Проверяет корректность пользовательской реализации ``square``."""

    assert square(3) == 9
    assert square(0) == 0
    assert square(-3) == 9
    assert square(1.5) == 2.25
    print("Task2 OK")


__all__ = ["run_tests", "Number"]
