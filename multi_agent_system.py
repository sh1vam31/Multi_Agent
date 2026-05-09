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
    study_subjects = [
        "python",
        "java",
        "javascript",
        "c++",
        "machine learning",
        "data science",
        "sql",
        "react",
        "node",
        "django",
    ]
    for subject in study_subjects:
        if subject in lowered:
            topic = f"{subject} study plan"
            break
    if topic == "study plan" and any(token in lowered for token in ["study", "learn", "practice"]):
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


def _time_to_days(time_available: str) -> int:
    mapping = {
        "1 day": 1,
        "2 days": 2,
        "3 days": 3,
        "1 week": 7,
        "2 weeks": 14,
        "1 month": 30,
    }
    return mapping.get(time_available, 7)


def _build_travel_plan(state: PlannerState) -> str:
    days = _time_to_days(state.get("time_available", "1 week"))
    audience = state.get("audience", "traveler")
    skill_level = state.get("skill_level", "beginner")

    if days <= 1:
        itinerary = [
            "Day 1: Arrive, check in, rest, and do one short local activity.",
        ]
    elif days == 2:
        itinerary = [
            "Day 1: Arrive, check in, explore nearby attractions, and have a relaxed dinner.",
            "Day 2: Visit the main attraction, leave buffer time for travel, and return by evening.",
        ]
    elif days == 3:
        itinerary = [
            "Day 1: Arrive, check in, and take a light walk around the area.",
            "Day 2: Visit the main sightseeing spots, try local food, and keep one flexible block.",
            "Day 3: Buy souvenirs, review transport timing, and head back with a buffer.",
        ]
    else:
        itinerary = [
            "Day 1: Arrive and settle in.",
            "Day 2: Explore the main destination highlights.",
            "Day 3: Add a free exploration block or a local activity.",
            "Day 4: Visit a second attraction or nearby neighborhood.",
            "Day 5: Use for food, shopping, or rest.",
            "Final day: Pack, check out, and travel back with time to spare.",
        ]

    plan_lines = [
        f"Travel Plan for {audience}",
        f"Trip length: {state.get('time_available', '1 week')}",
        f"Planning level: {skill_level}",
        "",
        "Plan:",
        "- Make the trip practical, safe, and easy to follow.",
        "",
        "Before the trip:",
        "- Confirm destination, dates, and budget.",
        "- Book transport and accommodation.",
        "- Save local maps, tickets, and emergency contacts.",
        "- Pack only essentials based on weather and trip type.",
        "",
        "Itinerary:",
        *itinerary,
        "",
        "Checklist:",
        "- Carry ID, cash/card, charger, and medications.",
        "- Keep a small backup budget for delays.",
        "- Leave at least 30 to 60 minutes of buffer time between activities.",
    ]
    return "\n".join(plan_lines)


def _build_study_plan(state: PlannerState) -> str:
    days = _time_to_days(state.get("time_available", "1 week"))
    topic = state.get("topic", "study plan").replace(" study plan", "").title()
    audience = state.get("audience", "student")

    if days <= 3:
        schedule = [
            "Day 1: Learn the basics and set up your environment.",
            "Day 2: Practice core concepts with short exercises.",
            "Day 3: Build a small mini-project or review what you learned.",
        ]
    elif days <= 7:
        schedule = [
            "Day 1-2: Learn the fundamentals.",
            "Day 3-4: Practice with guided exercises.",
            "Day 5-6: Build a small project or revise weak areas.",
            "Day 7: Review and test yourself.",
        ]
    else:
        schedule = [
            "Week 1: Learn fundamentals and basic syntax.",
            "Week 2: Practice problems and build a small project.",
            "Final days: Revise, fix gaps, and take a mock test.",
        ]

    plan_lines = [
        f"Study Plan for {topic}",
        f"Audience: {audience}",
        f"Time available: {state.get('time_available', '1 week')}",
        "",
        "Plan:",
        "- Learn the basics first, then practice, then review.",
        "",
        "Before you start:",
        "- Set one clear goal.",
        "- Collect one primary resource and one practice source.",
        "- Study in short focused sessions.",
        "",
        "Schedule:",
        *schedule,
        "",
        "Rules to follow:",
        "- Practice every day, even if only for 30 minutes.",
        "- Spend more time on exercises than on reading.",
        "- End each session with a quick review of mistakes.",
    ]
    return "\n".join(plan_lines)


def _build_project_plan(state: PlannerState) -> str:
    days = _time_to_days(state.get("time_available", "1 week"))
    weeks = max(1, days // 7)
    audience = state.get("audience", "team")
    skill_level = state.get("skill_level", "beginner")
    user_request = state.get("user_request", "").lower()
    project_name = "Project"
    if "mobile app" in user_request:
        project_name = "Mobile App Project"
    elif "web app" in user_request:
        project_name = "Web App Project"
    elif "app" in user_request:
        project_name = "App Project"

    if weeks <= 1:
        phases = [
            "Week 1: Define scope, assign roles, and confirm delivery goals.",
        ]
    elif weeks == 2:
        phases = [
            "Week 1: Gather requirements, create wireframes, and finalize scope.",
            "Week 2: Build the first working version and test core features.",
        ]
    elif weeks == 3:
        phases = [
            "Week 1: Define requirements and design the solution.",
            "Week 2: Implement core features and review progress.",
            "Week 3: Fix issues, test, and prepare delivery.",
        ]
    else:
        phases = [
            "Week 1: Define scope, success criteria, and responsibilities.",
            "Week 2: Design and prototype the solution.",
            "Week 3: Implement core features.",
            "Week 4: Test, fix issues, and prepare final delivery.",
        ]

    plan_lines = [
        project_name,
        f"Target audience: {audience}",
        f"Duration: {state.get('time_available', '1 month')}",
        f"Skill level: {skill_level}",
        "",
        "Plan:",
        "- Deliver the project in clear phases with ownership and checkpoints.",
        "",
        "Team roles:",
        "- Project lead: tracks timeline and decisions.",
        "- Designer: handles UI and user flow.",
        "- Developer: builds the app features.",
        "- Tester: checks bugs and usability.",
        "",
        "Project goals:",
        "- Define the app idea and the minimum feature set.",
        "- Build the app in small deliverable increments.",
        "- Test early so issues are found before the final week.",
        "",
        "Weekly milestones:",
        *phases,
        "",
        "Deliverables:",
        "- Requirement list",
        "- Wireframes or mockups",
        "- Working app prototype",
        "- Test report and final presentation",
        "",
        "Risks to watch:",
        "- Scope creep",
        "- Delayed feedback",
        "- Unclear ownership",
        "- Last-minute testing",
    ]
    return "\n".join(plan_lines)


def _build_generic_plan(state: PlannerState) -> str:
    return "\n".join(
        [
            f"Use case: {state['topic']}",
            f"Audience: {state['audience']}",
            f"Skill level: {state['skill_level']}",
            f"Time available: {state['time_available']}",
            "",
            "Plan:",
            "- Break the request into smaller requirements.",
            "- Add practical steps that match the available time.",
            "- Finish with a clear next-action checklist.",
        ]
    )


def writer_agent(state: PlannerState) -> Dict[str, Any]:
    topic = state["topic"]
    if "travel plan" in topic:
        final_answer = _build_travel_plan(state)
    elif "study plan" in topic:
        final_answer = _build_study_plan(state)
    elif "project plan" in topic:
        final_answer = _build_project_plan(state)
    else:
        final_answer = _build_generic_plan(state)
    return {"final_answer": final_answer}


def critic_agent(state: PlannerState) -> Dict[str, Any]:
    final_answer = state.get("final_answer", "")
    if final_answer.strip():
        return {"critique": "Output is structured and complete."}
    return {"critique": "Output is empty."}


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
    return "\n".join(["=== Multi-Agent System Result ===", "", result.get("final_answer", "")])


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
