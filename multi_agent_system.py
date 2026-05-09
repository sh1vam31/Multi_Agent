"""Multi-agent study planner built around a LangGraph-style workflow.

This file is intentionally self-contained so it can satisfy the assignment
requirements in environments where LangGraph/LangChain are unavailable.

If the optional dependencies are installed, the script uses them directly.
Otherwise it falls back to a small local graph runner with the same node
structure, state passing, and main() entrypoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict


try:  # Optional dependency support
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover - fallback path for the local environment
    END = "__end__"

    class StateGraph:  # Minimal compatible fallback
        def __init__(self, state_type: type):
            self.state_type = state_type
            self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
            self.edges: Dict[str, List[str]] = {}
            self.entry_point: Optional[str] = None

        def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
            self.nodes[name] = fn

        def add_edge(self, source: str, target: str):
            self.edges.setdefault(source, []).append(target)

        def set_entry_point(self, name: str):
            self.entry_point = name

        def compile(self):
            graph = self

            class _App:
                def invoke(self, state: Dict[str, Any]):
                    current_state = dict(state)
                    node = graph.entry_point
                    while node and node != END:
                        current_state = _merge_state(
                            current_state, graph.nodes[node](current_state)
                        )
                        next_nodes = graph.edges.get(node, [])
                        node = next_nodes[0] if next_nodes else END
                    return current_state

            return _App()


class PlannerState(TypedDict, total=False):
    user_request: str
    topic: str
    audience: str
    time_available: str
    skill_level: str
    goals: List[str]
    constraints: List[str]
    research_notes: List[str]
    plan: List[str]
    final_answer: str
    critique: str


def _merge_state(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state)
    for key, value in update.items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = merged[key] + value
        else:
            merged[key] = value
    return merged


def parse_request(user_request: str) -> PlannerState:
    lowered = user_request.lower()
    topic = "study plan"
    if "travel" in lowered:
        topic = "travel plan"
    elif "resume" in lowered:
        topic = "resume review"
    elif "slide" in lowered:
        topic = "slide outline"
    elif "recipe" in lowered:
        topic = "recipe idea"
    elif "content" in lowered:
        topic = "content strategy"
    elif "project" in lowered:
        topic = "project plan"

    skill_level = "beginner"
    if any(token in lowered for token in ["advanced", "expert", "senior"]):
        skill_level = "advanced"
    elif any(token in lowered for token in ["intermediate", "mid"]):
        skill_level = "intermediate"

    audience = "student"
    if "team" in lowered:
        audience = "team"
    elif "client" in lowered:
        audience = "client"

    time_available = "1 week"
    for candidate in ["1 day", "2 days", "3 days", "1 week", "2 weeks", "1 month"]:
        if candidate in lowered:
            time_available = candidate
            break

    return {
        "user_request": user_request,
        "topic": topic,
        "audience": audience,
        "time_available": time_available,
        "skill_level": skill_level,
    }


def requirements_analyst(state: PlannerState) -> Dict[str, Any]:
    goals = [
        f"Create a useful {state['topic']} tailored to the {state['skill_level']} level.",
        f"Keep the output appropriate for a {state['audience']}.",
        f"Respect the available time horizon of {state['time_available']}.",
    ]
    constraints = [
        "Use a clear workflow with multiple specialist agents.",
        "Pass shared context between nodes.",
        "Return an actionable, structured final result.",
    ]
    return {"goals": goals, "constraints": constraints}


def planning_agent(state: PlannerState) -> Dict[str, Any]:
    goals = state.get("goals", [])
    constraints = state.get("constraints", [])
    plan = [
        "1. Break the user request into task-specific requirements.",
        "2. Generate supporting notes for the chosen use case.",
        "3. Synthesize an output plan with clear sections and next steps.",
        "4. Review the draft for completeness and coherence.",
    ]
    if goals:
        plan.append(f"5. Align the final output with: {goals[0]}")
    if constraints:
        plan.append(f"6. Keep the workflow within: {constraints[0]}")
    return {"plan": plan}


def research_agent(state: PlannerState) -> Dict[str, Any]:
    topic = state["topic"]
    research_notes = [
        f"{topic.title()} should start with the user's end goal and success criteria.",
        "A multi-agent design works best when each agent owns one clear responsibility.",
        "Shared state should carry extracted requirements, intermediate notes, and the final draft.",
        "The final answer should be concrete enough that the user can act on it immediately.",
    ]
    return {"research_notes": research_notes}


def writer_agent(state: PlannerState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    research_notes = state.get("research_notes", [])
    goals = state.get("goals", [])
    body = [
        f"Use case: {state['topic']}",
        f"Audience: {state['audience']}",
        f"Skill level: {state['skill_level']}",
        f"Time available: {state['time_available']}",
        "",
        "Plan:",
        *plan,
        "",
        "Key notes:",
        *[f"- {note}" for note in research_notes],
    ]
    if goals:
        body.extend(["", "Primary goal:", f"- {goals[0]}"])
    return {"final_answer": "\n".join(body)}


def critic_agent(state: PlannerState) -> Dict[str, Any]:
    final_answer = state.get("final_answer", "")
    critique_parts = []
    if "Plan:" not in final_answer:
        critique_parts.append("Missing plan section.")
    if "Key notes:" not in final_answer:
        critique_parts.append("Missing research notes section.")
    if len(state.get("research_notes", [])) < 3:
        critique_parts.append("Research layer is too thin.")
    if not critique_parts:
        critique_parts.append("Output is structured and complete.")
    return {"critique": " ".join(critique_parts)}


def build_graph():
    graph = StateGraph(PlannerState)
    graph.add_node("requirements", requirements_analyst)
    graph.add_node("planner", planning_agent)
    graph.add_node("researcher", research_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)

    graph.set_entry_point("requirements")
    graph.add_edge("requirements", "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def run_system(user_request: str) -> PlannerState:
    state = parse_request(user_request)
    app = build_graph()
    result = app.invoke(state)
    return result  # type: ignore[return-value]


def format_output(result: PlannerState) -> str:
    return "\n".join(
        [
            "=== Multi-Agent System Result ===",
            "",
            result.get("final_answer", ""),
            "",
            "Critique:",
            result.get("critique", ""),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-agent LangGraph-style planner."
    )
    parser.add_argument(
        "--request",
        "-r",
        help="User request to analyze and process.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final state as JSON.",
    )
    args = parser.parse_args()

    user_request = args.request or input("Enter your request: ").strip()
    if not user_request:
        raise SystemExit("A request is required.")

    result = run_system(user_request)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_output(result))


if __name__ == "__main__":
    main()
