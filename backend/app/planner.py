"""Turn a natural-language instruction into a short tool sequence.

When the local model is on, it is the only thing that chooses which seen
object to use. The built-in word matcher runs only when the model is off.
The model still cannot add a tool this arm does not have, and it cannot
emit joint angles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from app.config import Config
from app.ollama import ollama_base
from app.vision import Observation

TOOLS = (
    "scan",
    "inspect",
    "locate",
    "approach",
    "grasp",
    "lift",
    "sample_odor",
    "place",
    "park",
    "report_scene",
)

TOOL_LABELS = {
    "scan": "Scan",
    "inspect": "Inspect",
    "locate": "Locate",
    "approach": "Approach",
    "grasp": "Grasp",
    "lift": "Lift",
    "sample_odor": "Sample odor",
    "place": "Place",
    "park": "Park",
    "report_scene": "Report scene",
}

UNSUPPORTED = (
    (r"\bthrow\b|\btoss\b", "throwing"),
    (r"\bpour\b", "pouring"),
    (r"\bshake\b", "shaking"),
    (r"\bwave\b", "gesturing"),
    (r"\bcut\b|\bslice\b", "cutting"),
    (r"\bunscrew\b|\bopen the (jar|bottle|lid)\b", "opening containers"),
    (r"\btemperatur", "temperature measurement"),
    (r"\bweigh\b", "weighing"),
    (r"\bdrive\b|\bnavigate\b", "driving a mobile base"),
    (r"\bfollow me\b", "following a person"),
)

TARGET_RE = re.compile(
    r"\b(orange|blue|green|red|yellow)\s+(fruit|cup|block|mug)\b"
    r"|\b(fruit|cup|block|mug)\b"
    r"|\b(orange|blue|green|red|yellow)\b",
    re.I,
)

STOP_RE = re.compile(r"\s*(stop|e-stop|estop|abort|halt)\s*[.!]?\s*", re.I)


@dataclass
class TargetSpec:
    color: str | None
    kind: str | None
    phrase: str


@dataclass
class PlanResult:
    status: str
    summary: str = ""
    final_state: str = ""
    steps: list[dict] = field(default_factory=list)
    targets: list[dict] = field(default_factory=list)
    question: str = ""
    options: list[dict] = field(default_factory=list)
    missing_capability: str = ""
    planner: str = "built-in"
    planner_note: str = ""
    enose_note: str = ""
    intent: str = ""


class Planner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def plan(
        self,
        text: str,
        observations: list[Observation],
        enose_mode: str,
        answers: list[dict],
    ) -> PlanResult:
        cleaned = text.strip()
        if not cleaned:
            return PlanResult(
                status="clarification",
                question="What should I do with the arm?",
                planner_note="The instruction was empty.",
            )
        if STOP_RE.fullmatch(cleaned):
            return PlanResult(status="stop", summary="Stop.", intent="stop")
        for pattern, capability in UNSUPPORTED:
            if re.search(pattern, cleaned, re.I):
                return PlanResult(
                    status="unavailable",
                    missing_capability=capability,
                    summary=(
                        f"I can't do that. {capability.capitalize()} is not one of the arm's tools. "
                        "I can scan, inspect, locate, approach, grasp, lift, sample odor, place, and park."
                    ),
                    intent="unsupported",
                )
        if self.cfg.llm_enabled:
            return self._decide(cleaned, observations, enose_mode, answers)
        low = cleaned.lower()
        if re.search(r"what('s| is) on the table|show me what|look at the table|scan the table", low):
            result = self._survey()
        else:
            wants_smell = bool(re.search(r"\b(smell|sniff|odou?r)\b", low))
            wants_pick = bool(re.search(r"\b(pick|grab|grasp|lift)\b", low))
            wants_find = bool(re.search(r"\bfind\b|\bshow me\b|\bwhere\b", low))
            put_back = bool(re.search(r"\bput (it|them|that) back\b", low))
            destination = None
            dest_match = re.search(r"\bon the (left|right|center|front)\b", low)
            if dest_match and re.search(r"\bput\b", low):
                destination = dest_match.group(1)
            specs = _extract_targets(low)
            if not specs and not (wants_smell or wants_pick or wants_find):
                return PlanResult(
                    status="clarification",
                    question="What should I do? I can find something, pick it up, smell it, or put it down.",
                    intent="unclear",
                )
            if not specs:
                return PlanResult(
                    status="clarification",
                    question="Which object should I use?",
                    intent="unclear",
                )
            resolved, clarification = _resolve(specs, observations, answers)
            if clarification is not None:
                clarification.intent = "clarify"
                return clarification
            multi = len(resolved) > 1
            # The gripper holds one object. A later sample implies putting the earlier one back.
            place_each = put_back or destination is not None or (multi and (wants_smell or wants_pick))
            if wants_smell or wants_pick:
                result = self._manipulate(
                    resolved,
                    smell=wants_smell,
                    place=place_each,
                    destination=destination,
                    put_back=put_back or (multi and destination is None),
                )
            elif wants_find:
                result = self._find(resolved)
            else:
                return PlanResult(
                    status="clarification",
                    question="Should I pick it up, smell it, or only look?",
                    intent="unclear",
                )
        result.enose_note = _enose_note(result, enose_mode)
        return result

    def plan_for_object(self, item: Observation, enose_mode: str) -> PlanResult:
        """The user picked this detection. Do not ask the model to choose another object."""
        bound = _bind(TargetSpec(color=None, kind=None, phrase=item.name), item)
        result = self._manipulate([bound], smell=True, place=True, destination=None, put_back=True)
        result.enose_note = _enose_note(result, enose_mode)
        result.planner = "selected"
        result.planner_note = f"You selected {item.name} from the camera list."
        return result

    def _survey(self) -> PlanResult:
        steps = [
            _step("scan", "Scan the taught tabletop viewpoints."),
            _step("inspect", "Read the latest frame."),
            _step("report_scene", "List what is visible, including anything uncertain."),
            _step("park", "Return to the rest pose."),
        ]
        return PlanResult(
            status="ready",
            intent="survey",
            summary="Scan the table and report what is visible. The arm will not grasp.",
            final_state="The arm is back at the rest pose. Nothing is picked up.",
            steps=steps,
        )

    def _find(self, resolved: list[dict]) -> PlanResult:
        steps = [_step("scan", "Scan the taught tabletop viewpoints.")]
        for index, item in enumerate(resolved):
            steps.append(_step("inspect", f"Look for the {item['phrase']}.", index, item))
            steps.append(_step("locate", f"Identify the {item['phrase']} and check whether it is reachable.", index, item))
        steps.append(_step("park", "Return to the rest pose."))
        names = _join(item["name"] for item in resolved)
        return PlanResult(
            status="ready",
            intent="find",
            summary=f"Look for {names}. I will not pick anything up.",
            final_state="The arm is back at the rest pose. Nothing is picked up.",
            steps=steps,
            targets=resolved,
        )

    def _manipulate(self, resolved, smell: bool, place: bool, destination: str | None, put_back: bool) -> PlanResult:
        steps = [_step("scan", "Start from the rest pose and scan the validated tabletop.")]
        for index, item in enumerate(resolved):
            name = item["name"]
            steps.append(_step("inspect", f"Take a fresh frame before acting on the {name}.", index, item))
            steps.append(_step("locate", f"Confirm the {name} and whether it is inside the pickup area.", index, item))
            steps.append(_step("approach", f"Approach the {name} with the gripper open.", index, item))
            steps.append(_step("grasp", f"Close the gripper to the calibrated angle for the {name}.", index, item))
            steps.append(_step("lift", f"Lift the {name} in short joint steps.", index, item))
            steps.append(
                _step(
                    "inspect",
                    f"Take a fresh frame and check whether the {name} moved with the gripper.",
                    index,
                    item,
                    purpose="confirm_pickup",
                )
            )
            if smell:
                steps.append(
                    _step(
                        "sample_odor",
                        f"Hold the {name} steady at the wrist inlet. Request a baseline, a sample, and a purge.",
                        index,
                        item,
                    )
                )
            if place:
                if put_back or destination is None:
                    place_detail = f"Put the {name} back where it started."
                    place_dest = "original"
                else:
                    place_detail = f"Put the {name} on the {destination}."
                    place_dest = destination
                steps.append(_step("place", place_detail, index, item, destination=place_dest))
        steps.append(_step("park", "Return to the rest pose."))
        names = _join(item["name"] for item in resolved)
        if smell and place:
            summary = f"Pick up {names}, sample at the wrist inlet, then put the object back."
        elif smell:
            summary = f"Pick up {names} and hold it at the wrist inlet for a sample."
        elif place:
            summary = f"Pick up {names} and place it."
        else:
            summary = f"Pick up {names}."
        final = _final_state(resolved, smell, place, destination, put_back)
        summary = _with_reach_notes(self.cfg, resolved, summary)
        return PlanResult(
            status="ready",
            intent="manipulate",
            summary=summary,
            final_state=final,
            steps=steps,
            targets=resolved,
        )

    def _decide(self, text: str, observations: list[Observation], enose_mode: str, answers: list[dict]) -> PlanResult:
        """The model picks the object and the steps. A failed reply is not replaced by word matching."""
        by_id = {}
        for item in observations:
            if item.held:
                continue
            by_id.setdefault(item.id, item)
        scene = [
            {"id": item.id, "name": item.name, "zone": item.zone, "uncertain": item.uncertain}
            for item in by_id.values()
        ]
        prompt = {
            "instruction": text,
            "seen": scene,
            "answers": answers,
            "allowed_tools": list(TOOLS),
            "reply": {
                "summary": "what you decided",
                "object_ids": ["an id from seen, or empty"],
                "steps": [{"tool": "locate", "detail": "why"}],
                "question": "",
            },
        }
        system = (
            "You choose which object in seen the user asked to use, and which tools to run. "
            "Decide from the user's words and the names in seen. "
            "Choose only ids from seen. Use only allowed tools, in the order they should run. "
            "Do not invent joint angles, serial commands, or objects that are not in seen. "
            "If none of the names is what the user asked for, set object_ids to [] and set question to a short sentence that names what you see. "
            "If more than one seen object could be the one they mean, set question and leave object_ids empty."
        )
        missed = PlanResult(
            status="clarification",
            question="The language model did not choose an object. Plan again.",
            options=[{"id": item.id, "label": item.name} for item in by_id.values()],
            planner="local-llm",
            intent="clarify",
        )
        base = self.cfg.llm_base_url.rstrip("/")
        try:
            if self.cfg.llm_api == "ollama":
                base = ollama_base(base)
                response = httpx.post(
                    base + "/api/chat",
                    json={
                        "model": self.cfg.llm_model,
                        "stream": False,
                        "think": False,
                        "format": "json",
                        "keep_alive": "30m",
                        "options": {"temperature": 0, "num_predict": 500, "num_ctx": 4096},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(prompt)},
                        ],
                    },
                    timeout=self.cfg.llm_timeout_s,
                )
            else:
                response = httpx.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": self.cfg.llm_model,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(prompt)},
                        ],
                    },
                    timeout=self.cfg.llm_timeout_s,
                )
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"] if self.cfg.llm_api == "ollama" else body["choices"][0]["message"]["content"]
            parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
        except Exception as exc:
            missed.planner_note = f"Local LLM was not used ({exc})."
            return missed
        adopted = _plan_from_llm(parsed, by_id, missed)
        if adopted.planner != "local-llm":
            missed.planner_note = adopted.planner_note or missed.question
            missed.question = missed.planner_note
            return missed
        if adopted.status == "ready":
            adopted.enose_note = _enose_note(adopted, enose_mode)
            adopted.summary = _with_reach_notes(self.cfg, adopted.targets, adopted.summary)
        return adopted

    def choose_seen(self, request: str, observations: list[Observation]) -> Observation | None:
        """Ask the model which fresh label is the object the user asked for."""
        if not self.cfg.llm_enabled:
            return None
        rows = [{"id": item.id, "name": item.name} for item in observations if not item.held]
        if not rows:
            return None
        system = (
            "The user named an object. Choose the id of the seen object that matches their request. "
            "Decide from their words and these names. "
            "Return JSON {\"id\":\"\"} using only an id from the list. "
            "Use an empty id if none of the names is what they asked for."
        )
        user = json.dumps({"request": request, "seen": rows})
        base = self.cfg.llm_base_url.rstrip("/")
        try:
            if self.cfg.llm_api == "ollama":
                response = httpx.post(
                    ollama_base(base) + "/api/chat",
                    json={
                        "model": self.cfg.llm_model,
                        "stream": False,
                        "think": False,
                        "format": "json",
                        "keep_alive": "30m",
                        "options": {"temperature": 0, "num_predict": 60, "num_ctx": 2048},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    timeout=self.cfg.llm_timeout_s,
                )
            else:
                response = httpx.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": self.cfg.llm_model,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    timeout=self.cfg.llm_timeout_s,
                )
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"] if self.cfg.llm_api == "ollama" else body["choices"][0]["message"]["content"]
            parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
        except Exception:
            return None
        chosen = next((item for item in observations if item.id == parsed.get("id")), None)
        return None if chosen is None or chosen.held else chosen


def _plan_from_llm(parsed: dict, by_id: dict, fallback: PlanResult) -> PlanResult:
    question = parsed.get("question")
    if isinstance(question, str) and question.strip() and not parsed.get("object_ids"):
        options = [
            {"id": item.id, "label": item.name}
            for item in by_id.values()
        ]
        if options:
            return PlanResult(
                status="clarification",
                question=question.strip(),
                options=options,
                planner="local-llm",
                planner_note="The local model asked which object you mean.",
            )
    steps_in = parsed.get("steps") if isinstance(parsed.get("steps"), list) else []
    tools = []
    for step in steps_in:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        if tool not in TOOLS:
            fallback.planner_note = (
                "The local model named a tool this arm does not have. The built-in sequence was kept."
            )
            return fallback
        tools.append(step)
    if not tools:
        fallback.planner_note = "The local model did not return a usable step list. The built-in sequence was kept."
        return fallback
    chosen = []
    for object_id in parsed.get("object_ids") or []:
        item = by_id.get(object_id)
        if item is not None and item not in chosen:
            chosen.append(item)
    targets = [_bind(TargetSpec(color=None, kind=None, phrase=item.name), item) for item in chosen]
    steps = []
    for step in tools:
        target = targets[0] if targets else None
        needs_object = step["tool"] not in {"scan", "park", "report_scene"}
        steps.append(
            _step(
                step["tool"],
                str(step.get("detail") or TOOL_LABELS[step["tool"]]).strip(),
                0 if needs_object and target else None,
                target if needs_object else None,
            )
        )
    summary = str(parsed.get("summary") or fallback.summary or "The local model chose this sequence.").strip()
    adopted = PlanResult(
        status="ready",
        summary=summary,
        final_state=summary,
        steps=steps,
        targets=targets,
        planner="local-llm",
        planner_note="The local model chose the object and the steps from what the camera saw.",
        intent="manipulate" if any(step["tool"] in {"grasp", "approach", "sample_odor"} for step in steps) else "survey",
    )
    if any(step["tool"] == "sample_odor" for step in adopted.steps):
        adopted.enose_note = fallback.enose_note
    return adopted


def _extract_targets(text: str) -> list[TargetSpec]:
    # Drop placement phrases so "left" is not an object and "put it on the left" adds no target.
    stripped = re.sub(r"\bon the (left|right|center|front)\b", " ", text)
    stripped = re.sub(r"\bput (it|them|that) back\b", " ", stripped)
    specs: list[TargetSpec] = []
    for match in TARGET_RE.finditer(stripped):
        if match.group(1) and match.group(2):
            color, kind = match.group(1).lower(), match.group(2).lower()
            phrase = f"{color} {kind}"
        elif match.group(3):
            color, kind = None, match.group(3).lower()
            phrase = kind
        else:
            color, kind = match.group(4).lower(), None
            phrase = color
        specs.append(TargetSpec(color=color, kind=kind, phrase=phrase))
    if not specs:
        for word in re.findall(r"\b(?:the|a|an)\s+([a-z]+)", stripped):
            if word in {"table", "arm", "robot", "gripper", "camera", "it", "them", "left", "right", "center", "front"}:
                continue
            specs.append(TargetSpec(color=None, kind=None, phrase=word))
    return _dedupe_overlapping(specs)


def _dedupe_overlapping(specs: list[TargetSpec]) -> list[TargetSpec]:
    kept: list[TargetSpec] = []
    for spec in specs:
        if any(spec.phrase != item.phrase and spec.phrase in item.phrase for item in specs):
            # "orange" inside an already-captured "orange fruit" span can still appear
            # when the regex backtracks across phrases. Drop the shorter duplicate
            # only when a longer phrase with the same color was found.
            if spec.kind is None and any(other.color == spec.color and other.kind for other in specs):
                continue
        kept.append(spec)
    return kept


def _resolve(specs: list[TargetSpec], observations: list[Observation], answers: list[dict]):
    resolved = []
    answer_queue = list(answers)
    for index, spec in enumerate(specs):
        matches = _match(spec, observations)
        if len(matches) <= 1:
            item = _bind(spec, matches[0] if matches else None)
            resolved.append(item)
            continue
        if not answer_queue:
            return [], _question(spec, matches)
        answer = answer_queue.pop(0)
        chosen = _choose(answer, matches)
        if chosen is None:
            question = _question(spec, matches)
            question.question = "I didn't catch which one. " + question.question
            return [], question
        resolved.append(_bind(spec, chosen))
    return resolved, None


def _words_fit(phrase: str, item: Observation) -> bool:
    """'bottle' matches a label such as 'Bottle of water'."""
    wanted = set(re.findall(r"[a-z0-9]+", phrase.lower()))
    if not wanted:
        return False
    have = set(
        re.findall(
            r"[a-z0-9]+",
            " ".join([item.name, item.label, item.kind, item.id.replace("_", " ")]).lower(),
        )
    )
    return wanted <= have


def _match(spec: TargetSpec, observations: list[Observation]) -> list[Observation]:
    hits = []
    for item in observations:
        if item.held:
            continue
        color_ok = spec.color is None or item.color == spec.color or item.label == spec.color
        kind_ok = spec.kind is None or item.kind == spec.kind or item.label == spec.kind
        if spec.color is None and spec.kind is None:
            if _words_fit(spec.phrase, item):
                hits.append(item)
            continue
        if spec.kind is None and spec.color:
            # "orange" matches an orange-colored object or one labeled orange.
            if item.color == spec.color or item.label == spec.color:
                hits.append(item)
            continue
        if color_ok and kind_ok:
            hits.append(item)
    return hits


def _choose(answer: dict, matches: list[Observation]) -> Observation | None:
    object_id = answer.get("object_id")
    if object_id:
        for item in matches:
            if item.id == object_id:
                return item
        return None
    text = (answer.get("text") or "").strip().lower()
    if not text:
        return None
    ranked = []
    for item in matches:
        score = 0
        if item.id.replace("_", " ") in text or item.name in text:
            score += 3
        if item.kind in text:
            score += 2
        if item.color in text:
            score += 1
        if item.label in text:
            score += 1
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if ranked and ranked[0][0] > 0 and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
        return ranked[0][1]
    return None


def _question(spec: TargetSpec, matches: list[Observation]) -> PlanResult:
    options = [
        {
            "id": item.id,
            "label": f"{item.name}" + (f", {item.zone}" if item.zone else "") + (", uncertain" if item.uncertain else ""),
        }
        for item in matches
    ]
    noun = spec.phrase
    return PlanResult(
        status="clarification",
        question=f"Which {noun} do you mean?",
        options=options,
        summary=f"I can see more than one match for “{noun}”.",
    )


def _bind(spec: TargetSpec, item: Observation | None) -> dict:
    if item is None:
        return {
            "object_id": None,
            "name": spec.phrase,
            "phrase": spec.phrase,
            "zone": None,
            "seen": False,
        }
    return {
        "object_id": item.id,
        "name": item.name,
        "phrase": spec.phrase,
        "zone": item.zone,
        "seen": True,
        "localizable": item.localizable,
    }


def _step(tool: str, detail: str, index: int | None = None, item: dict | None = None, purpose: str = "", destination: str | None = None) -> dict:
    return {
        "tool": tool,
        "label": TOOL_LABELS[tool],
        "detail": detail,
        "object_index": index,
        "object_id": None if item is None else item.get("object_id"),
        "description": None if item is None else item.get("phrase"),
        "purpose": purpose,
        "destination": destination,
    }


def _join(names) -> str:
    items = list(names)
    if not items:
        return "the object"
    if len(items) == 1:
        return f"the {items[0]}"
    return "the " + ", then the ".join(items)


def _final_state(resolved, smell, place, destination, put_back) -> str:
    sentences = []
    for item in resolved:
        name = item["name"]
        if place and (put_back or destination is None):
            where = item["zone"] or "its start zone"
            sentences.append(f"The {name} is back on the {where} zone.")
        elif place and destination:
            sentences.append(f"The {name} is on the {destination}.")
        elif smell:
            sentences.append(
                f"The arm parks while still holding the {name}, because the instruction did not say where to put it."
            )
        else:
            sentences.append(f"The arm parks while still holding the {name}.")
    if not any("holding" in sentence for sentence in sentences):
        sentences.append("The arm is at the rest pose with the gripper open.")
    return " ".join(sentences)


def _with_reach_notes(cfg: Config, resolved: list[dict], summary: str) -> str:
    notes = []
    for item in resolved:
        if not item.get("seen"):
            notes.append(f"I don't see a {item['name']} in the current frame. The scan will look again.")
            continue
        if item.get("localizable") is False or not cfg.zones.get(item.get("zone") or ""):
            notes.append(
                f"The {item['name']} has no taught table position. The arm will aim at it within its joint limits."
            )
    if not notes:
        return summary
    return summary + " " + " ".join(notes)


def _enose_note(result: PlanResult, enose_mode: str) -> str:
    if not any(step["tool"] == "sample_odor" for step in result.steps):
        return ""
    if enose_mode == "mock":
        return "E-nose is mocked. The sample will be labeled as a mock sensor result."
    if enose_mode == "unavailable":
        return "E-nose is not connected. Sampling will report that it could not be performed."
    return "E-nose adapter is set to hardware."
