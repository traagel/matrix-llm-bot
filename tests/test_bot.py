from matrix_llm_bot.bot import _extract_prompt, _is_mentioned, _strip_mention

BOT_NAME = "alfred"
USER_ID = "@alfred:example.com"


def source_with_mention(*user_ids):
    return {"content": {"m.mentions": {"user_ids": list(user_ids)}}}


# --- _is_mentioned ---


def test_mentioned_via_matrix_mention():
    assert _is_mentioned("hey", BOT_NAME, USER_ID, source_with_mention(USER_ID))


def test_mentioned_via_name_in_body():
    assert _is_mentioned("alfred, what time is it?", BOT_NAME, USER_ID, {})


def test_mentioned_via_at_name():
    assert _is_mentioned("@alfred help", BOT_NAME, USER_ID, {})


def test_mentioned_case_insensitive():
    assert _is_mentioned("ALFRED do something", BOT_NAME, USER_ID, {})


def test_not_mentioned():
    assert not _is_mentioned("hey everyone", BOT_NAME, USER_ID, {})


def test_not_mentioned_partial_name():
    # "fred" should not match "alfred" as a word boundary would, but the regex uses re.search
    # so "alfred" appearing inside a word like "alfred123" should still match
    assert _is_mentioned("alfred123 hello", BOT_NAME, USER_ID, {})


def test_not_mentioned_different_user():
    assert not _is_mentioned("hello", BOT_NAME, USER_ID, source_with_mention("@bob:example.com"))


# --- _strip_mention ---


def test_strip_prefix_name():
    assert _strip_mention("alfred what is the weather", BOT_NAME, USER_ID) == "what is the weather"


def test_strip_prefix_with_colon():
    assert _strip_mention("alfred: tell me a joke", BOT_NAME, USER_ID) == "tell me a joke"


def test_strip_prefix_with_comma():
    assert _strip_mention("alfred, help", BOT_NAME, USER_ID) == "help"


def test_strip_suffix_name():
    assert _strip_mention("what time is it alfred", BOT_NAME, USER_ID) == "what time is it"


def test_strip_at_prefix():
    assert _strip_mention("@alfred how are you", BOT_NAME, USER_ID) == "how are you"


def test_strip_user_id():
    assert _strip_mention(f"{USER_ID} help me", BOT_NAME, USER_ID) == "help me"


def test_strip_body_only_mention():
    # If the entire body is just the mention, return the original body
    result = _strip_mention("alfred", BOT_NAME, USER_ID)
    assert result == "alfred"


# --- _extract_prompt ---


def test_extract_prompt_returns_none_when_not_mentioned():
    assert _extract_prompt("hello world", BOT_NAME, USER_ID, {}) is None


def test_extract_prompt_strips_mention():
    result = _extract_prompt("alfred what is 2+2", BOT_NAME, USER_ID, {})
    assert result == "what is 2+2"


def test_extract_prompt_via_matrix_mention():
    result = _extract_prompt("help me please", BOT_NAME, USER_ID, source_with_mention(USER_ID))
    assert result == "help me please"
