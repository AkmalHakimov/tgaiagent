from agent.policy import clip_reply, enforce_policy, looks_like_question


def test_looks_like_question():
    assert looks_like_question("How are you?")
    assert looks_like_question("could you explain this")
    assert not looks_like_question("hello there")


def test_policy_blocks_high_risk():
    decision = enforce_policy("How to hack wifi password?", max_chars=400)
    assert decision.allowed is False
    assert decision.reason == "high_risk_content"


def test_reply_clipping():
    text = "a" * 40
    clipped = clip_reply(text, 10)
    assert clipped == "aaaaaaa..."
