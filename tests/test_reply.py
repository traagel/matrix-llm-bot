import pytest

from matrix_llm_bot.handlers.reply import ConversationHistory, _explicit_search_request, _is_k8s_query

# --- _is_k8s_query ---


def test_k8s_keyword_match():
    assert _is_k8s_query("is the cluster healthy?", ["cluster", "pod"], [], {})


def test_k8s_service_match():
    assert _is_k8s_query("is nginx running?", [], ["nginx"], {})


def test_k8s_alias_match():
    assert _is_k8s_query("check myapp status", [], [], {"myapp": "my-application"})


def test_k8s_no_match():
    assert not _is_k8s_query("what is the weather today?", ["cluster", "pod"], [], {})


def test_k8s_case_insensitive():
    assert _is_k8s_query("Check K8S status", ["k8s"], [], {})


def test_k8s_empty_inputs():
    assert not _is_k8s_query("anything", [], [], {})


# --- _explicit_search_request ---


@pytest.mark.parametrize(
    "prompt",
    [
        "search for the latest news",
        "look it up",
        "look up the price",
        "use the internet to find it",
        "google that for me",
        "find out what happened",
        "check online",
        "browse for results",
    ],
)
def test_explicit_search_detected(prompt):
    assert _explicit_search_request(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "what is 2 + 2",
        "tell me a joke",
        "how are you",
        "what did I say earlier",
    ],
)
def test_explicit_search_not_detected(prompt):
    assert not _explicit_search_request(prompt)


# --- ConversationHistory ---


def test_history_stores_messages():
    h = ConversationHistory(maxlen=10)
    dq = h.get("room1", "user1")
    dq.append({"role": "user", "content": "hello"})
    assert len(dq) == 1


def test_history_respects_maxlen():
    h = ConversationHistory(maxlen=2)
    dq = h.get("room1", "user1")
    for i in range(5):
        dq.append({"role": "user", "content": str(i)})
    assert len(dq) == 2
    assert list(dq)[-1]["content"] == "4"


def test_history_maxlen_zero_is_unlimited():
    h = ConversationHistory(maxlen=0)
    dq = h.get("room1", "user1")
    for i in range(100):
        dq.append({"role": "user", "content": str(i)})
    assert len(dq) == 100


def test_history_separate_per_user():
    h = ConversationHistory(maxlen=10)
    h.get("room1", "alice").append({"role": "user", "content": "hi"})
    assert len(h.get("room1", "bob")) == 0


def test_history_separate_per_room():
    h = ConversationHistory(maxlen=10)
    h.get("room1", "alice").append({"role": "user", "content": "hi"})
    assert len(h.get("room2", "alice")) == 0


def test_history_clear():
    h = ConversationHistory(maxlen=10)
    h.get("room1", "user1").append({"role": "user", "content": "hello"})
    h.clear("room1", "user1")
    assert len(h.get("room1", "user1")) == 0


def test_history_clear_nonexistent_is_noop():
    h = ConversationHistory(maxlen=10)
    h.clear("room1", "nobody")  # should not raise
