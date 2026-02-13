from __future__ import annotations

PLANNER_SYSTEM_PROMPT = """
You are an intent planner for a Telegram assistant.
Return strict JSON with keys:
- should_reply (bool)
- intent (string)
- confidence (number 0..1)
- reply_style (string)
- tool_calls (array of {name:string,args:object})
- rationale (string)

Rules:
- Reply only when the user asks a direct question or requests help.
- Keep confidence realistic.
- Use tools only when needed.
- Never hallucinate tool names. Allowed tools: now_time, calculator, recall_user_profile.
""".strip()


def build_planner_user_prompt(
    agent_name: str,
    sender_name: str,
    text: str,
    context_lines: list[str],
) -> str:
    context_block = "\n".join(context_lines) if context_lines else "(none)"
    return (
        f"Agent: {agent_name}\n"
        f"Sender: {sender_name}\n"
        f"Message: {text}\n"
        f"Recent context:\n{context_block}\n"
    )


def build_response_system_prompt(agent_name: str) -> str:
    return (
        f"You are {agent_name}, an autonomous Telegram assistant. "
        "Be concise, accurate, and practical. "
        "If unsure, say what is uncertain. "
        "Do not claim actions you did not perform."
    )


def build_response_user_prompt(
    message_text: str,
    context_lines: list[str],
    tool_outputs: list[str],
    profile_facts: list[str],
    intent: str,
    style: str,
) -> str:
    context_block = "\n".join(context_lines) if context_lines else "(none)"
    tool_block = "\n".join(tool_outputs) if tool_outputs else "(none)"
    profile_block = "\n".join(profile_facts) if profile_facts else "(none)"

    return (
        f"User message:\n{message_text}\n\n"
        f"Detected intent: {intent}\n"
        f"Reply style: {style}\n\n"
        f"Profile facts:\n{profile_block}\n\n"
        f"Recent context:\n{context_block}\n\n"
        f"Tool outputs:\n{tool_block}\n\n"
        "Write the best direct answer for Telegram."
    )
