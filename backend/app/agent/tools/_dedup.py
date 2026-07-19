"""Duplicate-call guard shared by the agent's search tools.

The system prompt tells the model to "never call the same tool twice with a
similar query", but the local 8B model doesn't reliably follow that on its
own — left unchecked it can re-issue an identical search several times in a
row, burning the tool-call budget until the graph hits its recursion limit.
This enforces the rule in code instead of relying purely on the prompt.
"""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

DUPLICATE_QUERY_MESSAGE = (
    "You already searched for this exact query earlier in this turn — do not "
    "search again. Use the results you already have, or write the final "
    "prompt now."
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_repeat_query(
    messages: list[AnyMessage],
    tool_name: str,
    query: str,
    exclude_call_id: str,
) -> bool:
    """True if `tool_name` was already called with an equivalent `query`
    earlier in the current turn (i.e. since the most recent HumanMessage).

    When a single AIMessage requests several identical calls in parallel,
    only calls that precede `exclude_call_id` in that message count as
    "earlier" — otherwise every one of them would see the others as a prior
    duplicate and all would be blocked, leaving none to actually run.
    """
    normalized = _normalize(query)
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, AIMessage):
            continue
        calls = message.tool_calls
        self_index = next(
            (i for i, call in enumerate(calls) if call.get("id") == exclude_call_id), None
        )
        relevant_calls = calls[:self_index] if self_index is not None else calls
        for call in relevant_calls:
            if call.get("name") != tool_name:
                continue
            if _normalize(str(call.get("args", {}).get("query", ""))) == normalized:
                return True
    return False
