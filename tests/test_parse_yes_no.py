import pytest

from matrix_llm_bot.integrations.ollama import _parse_yes_no


@pytest.mark.parametrize(
    "answer, default, expected",
    [
        ("YES", False, True),
        ("NO", True, False),
        ("yes", False, True),
        ("no", True, False),
        ("YES NO", False, True),  # YES appears first
        ("NO YES", False, False),  # NO appears first
        ("maybe", False, False),  # ambiguous → default False
        ("maybe", True, True),  # ambiguous → default True
        ("  YES  ", False, True),  # leading/trailing whitespace
        ("I think YES is correct", False, True),
        ("Absolutely NO", False, False),
    ],
)
def test_parse_yes_no(answer, default, expected):
    assert _parse_yes_no(answer, default) is expected
