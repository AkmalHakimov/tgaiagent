import pytest

from agent.tools import safe_calculate


def test_safe_calculate_basic_math():
    assert safe_calculate("2 + 2 * 5") == 12.0


def test_safe_calculate_blocks_names():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('whoami')")
