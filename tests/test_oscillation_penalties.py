"""Tests for reversible oscillation detection and penalties.

TODO: extend these heuristics if future games expose richer inverse-action patterns.
"""

from __future__ import annotations

from zork_agent.types import TextGameState, evaluate_reversible_action_loop


def _state(
    observation: str,
    *,
    inventory_text: str = "",
    valid_actions: list[str] | None = None,
    score: int = 0,
) -> TextGameState:
    """Build a compact state fixture for loop-heuristic tests."""

    return TextGameState(
        observation=observation,
        inventory_text=inventory_text,
        valid_actions=list(valid_actions or []),
        score=score,
    )


def test_open_close_abab_is_penalized_as_reversible_no_progress_loop() -> None:
    """Alternating open/close loops with no gain should incur all loop penalties."""

    closed_mailbox = _state(
        "You are standing next to a small mailbox.",
        valid_actions=["open mailbox", "look"],
    )
    open_mailbox = _state(
        "Opening the small mailbox reveals a leaflet.",
        valid_actions=["close mailbox", "read leaflet", "look"],
    )
    result = evaluate_reversible_action_loop(
        action="close mailbox",
        recent_actions=["open mailbox", "close mailbox", "open mailbox"],
        state_history=[closed_mailbox, open_mailbox, closed_mailbox, open_mailbox],
        current_state=closed_mailbox,
        immediate_inverse_penalty=0.75,
        repeated_pair_penalty=1.0,
        reversible_no_progress_penalty=1.0,
    )

    assert result.loop_detected is True
    assert result.inverse_of_previous is True
    assert result.repeated_pair_count == 1
    assert result.no_progress is True
    assert result.total_penalty == 2.75


def test_take_drop_with_no_gain_is_penalized() -> None:
    """Immediate take/drop reversals should be penalized when they return to the same state."""

    empty_hands = _state(
        "A brass lamp lies here.",
        inventory_text="You are empty-handed.",
        valid_actions=["take lamp", "look"],
    )
    holding_lamp = _state(
        "You are holding a brass lamp.",
        inventory_text="brass lamp",
        valid_actions=["drop lamp", "look"],
    )
    result = evaluate_reversible_action_loop(
        action="drop lamp",
        recent_actions=["take lamp"],
        state_history=[empty_hands, holding_lamp],
        current_state=empty_hands,
        immediate_inverse_penalty=0.75,
        repeated_pair_penalty=1.0,
        reversible_no_progress_penalty=1.0,
    )

    assert result.loop_detected is True
    assert result.inverse_of_previous is True
    assert result.repeated_pair_count == 0
    assert result.no_progress is True
    assert result.total_penalty == 1.75


def test_reversal_with_real_score_gain_is_not_penalized() -> None:
    """Inverse actions should not be penalized if the cycle preserved a real score gain."""

    before_take = _state(
        "A jeweled egg rests here.",
        inventory_text="You are empty-handed.",
        valid_actions=["take egg", "look"],
        score=0,
    )
    after_take = _state(
        "You are carrying the jeweled egg.",
        inventory_text="jeweled egg",
        valid_actions=["drop egg", "look"],
        score=1,
    )
    after_drop = _state(
        "The jeweled egg is back on the ground.",
        inventory_text="You are empty-handed.",
        valid_actions=["take egg", "look"],
        score=1,
    )
    result = evaluate_reversible_action_loop(
        action="drop egg",
        recent_actions=["take egg"],
        state_history=[before_take, after_take],
        current_state=after_drop,
        immediate_inverse_penalty=0.75,
        repeated_pair_penalty=1.0,
        reversible_no_progress_penalty=1.0,
    )

    assert result.loop_detected is False
    assert result.inverse_of_previous is True
    assert result.no_progress is False
    assert result.total_penalty == 0.0
