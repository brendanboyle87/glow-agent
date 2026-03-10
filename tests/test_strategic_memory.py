"""Tests for map-memory-driven strategic guidance.

TODO: add longer-horizon route-memory tests if strategic planning becomes more explicit.
"""

from __future__ import annotations

from zork_agent.memory.frontier import FrontierEntry
from zork_agent.types import (
    EpisodeMapMemory,
    RegionNodeMemory,
    RegionOpportunity,
    RegionOpportunityKind,
    StrategicMode,
    TextGameState,
    derive_state_cluster_id,
)


def test_strategic_guidance_prefers_structural_frontier_over_harvested_local_followups() -> None:
    """A harvested local basin should yield to a stronger structural frontier opportunity."""

    memory = EpisodeMapMemory()
    memory.nodes["tree"] = RegionNodeMemory(
        cluster_id="tree",
        visit_count=5,
        durable_progress_count=1,
        opportunity_ids={"object:tree:egg"},
    )
    memory.opportunities["object:tree:egg"] = RegionOpportunity(
        opportunity_id="object:tree:egg",
        cluster_id="tree",
        kind=RegionOpportunityKind.OBJECT_FOLLOWUP,
        token="egg",
        action_hints=["open egg", "take egg"],
        no_progress_attempt_count=1,
    )
    memory.nodes["house"] = RegionNodeMemory(
        cluster_id="house",
        visit_count=1,
        durable_progress_count=0,
        opportunity_ids={"structural:house:window"},
    )
    memory.opportunities["structural:house:window"] = RegionOpportunity(
        opportunity_id="structural:house:window",
        cluster_id="house",
        kind=RegionOpportunityKind.STRUCTURAL_ACCESS,
        token="window",
        action_hints=["open window", "enter window"],
        base_priority=2.0,
    )

    guidance = memory.recommend_guidance(
        current_state=TextGameState(
            observation="You are at the tree.",
            valid_actions=["open egg", "take egg", "south"],
            state_cluster_id="tree",
            world_state_hash="tree",
        ),
        frontier_entries=[
            FrontierEntry(
                state_id="house-entry",
                score=0.0,
                depth=2,
                state_cluster_id="house",
                world_state_hash="house",
                summary_text="West of House by the openable window.",
            )
        ],
    )

    assert guidance.mode is StrategicMode.EXPLOIT
    assert guidance.preferred_cluster_id == "house"
    assert "structural" in guidance.reason.lower() or "harvested" in guidance.reason.lower()
    assert "open egg" in guidance.avoid_actions


def test_strategic_guidance_prefers_exits_over_local_object_churn_in_harvested_region() -> None:
    """A harvested region with only object followups should prefer exits and avoid local churn."""

    memory = EpisodeMapMemory()
    memory.nodes["tree"] = RegionNodeMemory(
        cluster_id="tree",
        visit_count=4,
        durable_progress_count=1,
        exit_actions={"south", "north"},
        explored_exit_actions={"north"},
        opportunity_ids={"object:tree:nest", "exit:tree:south"},
    )
    memory.opportunities["object:tree:nest"] = RegionOpportunity(
        opportunity_id="object:tree:nest",
        cluster_id="tree",
        kind=RegionOpportunityKind.OBJECT_FOLLOWUP,
        token="nest",
        action_hints=["take nest", "open nest"],
        no_progress_attempt_count=1,
    )
    memory.opportunities["exit:tree:south"] = RegionOpportunity(
        opportunity_id="exit:tree:south",
        cluster_id="tree",
        kind=RegionOpportunityKind.UNEXPLORED_EXIT,
        token="south",
        action_hints=["south"],
        base_priority=1.0,
    )

    guidance = memory.recommend_guidance(
        current_state=TextGameState(
            observation="You are in the tree.",
            valid_actions=["take nest", "open nest", "south", "north"],
            state_cluster_id="tree",
            world_state_hash="tree",
        ),
        frontier_entries=[],
    )

    assert guidance.mode is StrategicMode.EXPLORE
    assert guidance.try_actions == ["south"]
    assert "take nest" in guidance.avoid_actions


def test_structural_access_is_not_resolved_by_affordance_gain_alone() -> None:
    """Opening a structure should not clear the strategic goal until the region actually changes."""

    memory = EpisodeMapMemory()
    opportunity = RegionOpportunity(
        opportunity_id="structural:house:window",
        cluster_id="house",
        kind=RegionOpportunityKind.STRUCTURAL_ACCESS,
        token="window",
        action_hints=["open window", "enter window"],
    )

    resolved = memory._opportunity_is_resolved(  # type: ignore[attr-defined]
        opportunity=opportunity,
        previous_state=TextGameState(
            observation="West of House.",
            valid_actions=["open window"],
            state_cluster_id="house",
        ),
        current_state=TextGameState(
            observation="The window is now open.",
            valid_actions=["enter window"],
            state_cluster_id="house",
        ),
        action="open window",
        score_gain=0,
        inventory_gain_count=0,
        affordance_gain=1,
    )

    assert resolved is False


def test_state_cluster_prefers_room_title_over_local_object_nouns() -> None:
    """Region clustering should stay anchored to room/location text instead of transient objects."""

    cluster_id = derive_state_cluster_id(
        observation="Up a Tree\nYou are about 10 feet above the ground amidst the tree branches.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["take egg", "close nest", "down"],
    )

    assert "tree" in cluster_id
    assert "leaflet" not in cluster_id


def test_state_cluster_prefers_house_location_over_window_object() -> None:
    """Structural objects like windows should not replace the surrounding house region label."""

    cluster_id = derive_state_cluster_id(
        observation="Behind House\nYou are behind the white house. There is an open window here.",
        valid_actions=["open window", "enter window", "east", "west"],
    )

    assert "house" in cluster_id or "behind-house" in cluster_id


def test_state_cluster_does_not_treat_parser_failure_sentence_as_room_title() -> None:
    """Parser failure sentences should not be promoted into synthetic room-title clusters."""

    cluster_id = derive_state_cluster_id(
        observation="The rank undergrowth prevents eastward movement.",
        valid_actions=["east", "west", "south"],
    )

    assert cluster_id != "region:title:the-rank-undergrowth-prevents-eastward-movement"
