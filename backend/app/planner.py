"""Turn a natural-language instruction into a short tool sequence.

When the local model is on, it only chooses which seen object to use.
The step list is always the arm's fixed sequence. The built-in word
matcher runs only when the model is off.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

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
        """The model only names the object. The step list is the fixed arm sequence."""
        by_id = {}
        for item in observations:
            if item.held:
                continue
            by_id.setdefault(item.id, item)
        width = _image_width(self.cfg)
        height = _image_height(self.cfg)
        order = _from_left(list(by_id.values()), width)
        scene = []
        for item in by_id.values():
            box = list(item.bbox or []) + [0, 0, 0, 0]
            u, v, w, h = box[:4]
            scene.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "kind": item.kind or item.label,
                    "place": _place_label(_across(item, width)),
                    "across": round(_across(item, width), 2),
                    "depth": round(_depth(item, height), 2),
                    "cx": round(float(u) + float(w) / 2),
                    "cy": round(float(v) + float(h) / 2),
                    "from_left": order[item.id],
                    "zone": item.zone,
                    "uncertain": item.uncertain,
                }
            )
        prompt = {
            "instruction": text,
            "seen": scene,
            "answers": answers,
        }
        system = (
            "Choose the id of the object the user asked to pick. "
            "The arm already has a fixed sequence of steps. Do not choose tools or steps. "
            "kind is the detector name. A glass is a cup, or a name containing glass. A bottle is not a glass. "
            "across is 0 at the left edge and 1 at the right edge. depth is 0 near the bottom and 1 far at the top. "
            "cx and cy are the picture center of the object. from_left counts the same kind from the left, starting at 1. "
            "The kind the user says to pick is the object to return. A later object is only a landmark. Do not return the landmark id. "
            "If they only named a mouse, return a mouse. "
            "The furthest landmark is the one of that kind with the greatest depth. Nearest is the smallest distance between cx and cy. "
            "Return only one JSON object: "
            '{"id":"an id copied from seen","question":""}. '
            "If none matches, leave id empty and set question to a short sentence."
        )
        missed = PlanResult(
            status="clarification",
            question="The language model did not answer. Plan again.",
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
                        "options": {"temperature": 0, "num_predict": 512, "num_ctx": 4096},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(prompt)},
                        ],
                    },
                    timeout=max(self.cfg.llm_timeout_s, 45),
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
                    timeout=max(self.cfg.llm_timeout_s, 45),
                )
            response.raise_for_status()
            body = response.json()
            content = _choice_text(body, self.cfg.llm_api)
            log.info("planner model reply: %s", content[:300])
            parsed = _choice_json(content)
        except Exception as exc:
            log.warning("planner reply was not usable: %s", exc)
            missed.planner_note = f"Local LLM was not used ({exc})."
            return missed
        chosen = _object_from_llm(parsed, by_id)
        if isinstance(chosen, PlanResult):
            log.info("planner left the choice open: %s", json.dumps(parsed)[:500])
            return chosen
        result = self._fixed_sequence(text, chosen)
        result.planner = "local-llm"
        result.planner_note = f"The model chose {chosen.name}. The arm follows its fixed step sequence."
        result.enose_note = _enose_note(result, enose_mode)
        return result

    def _fixed_sequence(self, text: str, item: Observation) -> PlanResult:
        """Pick, smell, and put the object back where the gripper closed."""
        bound = _bind(TargetSpec(color=None, kind=None, phrase=item.name), item)
        low = text.lower()
        wants_pick = bool(re.search(r"\b(pick|grab|grasp|lift)\b", low))
        wants_smell = bool(re.search(r"\b(smell|sniff|odou?r)\b", low))
        wants_find = bool(re.search(r"\bfind\b|\bshow me\b|\bwhere\b", low))
        if wants_find and not (wants_pick or wants_smell):
            return self._find([bound])
        return self._manipulate(
            [bound],
            smell=True,
            place=True,
            destination=None,
            put_back=True,
        )

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
                        "options": {"temperature": 0, "num_predict": 512, "num_ctx": 2048},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    timeout=max(self.cfg.llm_timeout_s, 45),
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
                    timeout=max(self.cfg.llm_timeout_s, 45),
                )
            response.raise_for_status()
            body = response.json()
            parsed = _choice_json(_choice_text(body, self.cfg.llm_api))
        except Exception:
            return None
        chosen = next((item for item in observations if item.id == parsed.get("id")), None)
        return None if chosen is None or chosen.held else chosen


def _image_width(cfg: Config) -> float:
    return float((cfg.calibration or {}).get("image_width") or 640) or 640


def _image_height(cfg: Config) -> float:
    return float((cfg.calibration or {}).get("image_height") or 480) or 480


def _across(item: Observation, width: float) -> float:
    box = list(item.bbox or []) + [0, 0, 0, 0]
    u, _, w, _ = box[:4]
    return min(1.0, max(0.0, (float(u) + float(w) / 2) / width))


def _place_label(across: float) -> str:
    if across < 0.4:
        return "on the left"
    if across > 0.6:
        return "on the right"
    return "in the middle"


def _from_left(items: list[Observation], width: float) -> dict[str, int]:
    groups: dict[str, list[Observation]] = {}
    for item in items:
        kind = (item.kind or item.label or item.id).strip().lower()
        groups.setdefault(kind, []).append(item)
    labels: dict[str, int] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda item: _across(item, width))
        for index, item in enumerate(ordered, start=1):
            labels[item.id] = index
    return labels


def _depth(item: Observation, height: float) -> float:
    """Higher in the picture is farther. 1 is the top edge and 0 is the bottom."""
    box = list(item.bbox or []) + [0, 0, 0, 0]
    _, v, _, h = box[:4]
    down = (float(v) + float(h) / 2) / height
    return 1.0 - min(1.0, max(0.0, down))


def _choice_text(body: dict, api: str) -> str:
    """gpt-oss can spend the token budget on thinking and leave content empty."""
    if api == "ollama":
        message = body.get("message") or {}
        content = str(message.get("content") or "")
        if "{" in content:
            return content
        thinking = str(message.get("thinking") or "")
        if "{" in thinking:
            return thinking
        return content
    message = ((body.get("choices") or [{}])[0].get("message") or {})
    return str(message.get("content") or "")


def _choice_json(text: str) -> dict:
    """Read one JSON object. A later id object wins when the first one is only the request."""
    text = text or ""
    decoder = json.JSONDecoder()
    first = text.find("{")
    if first < 0:
        raise json.JSONDecodeError("no object", text, 0)
    parsed, _ = decoder.raw_decode(text[first:])
    if isinstance(parsed, dict) and (
        _decision_useful(parsed)
        or any(isinstance(parsed.get(key), dict) for key in ("reply", "response", "plan", "result"))
    ):
        return parsed
    found = list(re.finditer(r"\{\s*\"id\"", text))
    if found:
        choice, _ = decoder.raw_decode(text[found[-1].start() :])
        if isinstance(choice, dict):
            return choice
    if isinstance(parsed, dict):
        return parsed
    raise json.JSONDecodeError("no object", text, first)


def _decision(parsed: dict) -> dict:
    """The model sometimes echoes the request and hides its choice under reply."""
    if not isinstance(parsed, dict):
        return {}
    nested = None
    for key in ("reply", "response", "plan", "result"):
        inner = parsed.get(key)
        if isinstance(inner, dict):
            nested = inner
            break
    if _decision_useful(parsed):
        return parsed
    return nested or parsed


def _decision_useful(obj: dict) -> bool:
    if obj.get("id") or obj.get("object_ids") or obj.get("object_id") or obj.get("objects"):
        return True
    question = obj.get("question")
    return isinstance(question, str) and bool(question.strip())


def _id_tokens(decision: dict) -> list[str]:
    raw = None
    for key in ("id", "object_ids", "object_id", "objects"):
        candidate = decision.get(key)
        if candidate is None or candidate == "" or candidate == []:
            continue
        raw = candidate
        break
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    tokens = []
    for item in raw:
        if isinstance(item, dict):
            token = item.get("id") or item.get("object_id") or item.get("name")
        else:
            token = item
        text = str(token or "").strip()
        if not text or "an id from seen" in text.lower():
            continue
        tokens.append(text)
    return tokens


def _match_token(token: str, items: list[Observation]):
    exact = [item for item in items if item.id == token]
    if exact:
        return exact[0]
    low = token.lower()
    named = [item for item in items if item.name.lower() == low or item.label.lower() == low]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        return "ambiguous"
    words = [word for word in re.findall(r"[a-z0-9]+", low) if len(word) > 2]
    if not words:
        return None
    partial = [item for item in items if all(word in item.name.lower() for word in words)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        return "ambiguous"
    return None


def _seen_sentence(items: list[Observation]) -> str:
    names = ", ".join(item.name for item in items)
    if not names:
        return "The camera did not name an object. Look again."
    return f"I see {names}. The model did not choose one."


def _clarify(question: str, items: list[Observation]) -> PlanResult:
    return PlanResult(
        status="clarification",
        question=question,
        options=[{"id": item.id, "label": item.name} for item in items],
        planner="local-llm",
        planner_note="The local model did not return a single object to use.",
        intent="clarify",
    )


def _object_from_llm(parsed: dict, by_id: dict) -> Observation | PlanResult:
    """The model's only job is to name one seen object. Its steps are ignored."""
    decision = _decision(parsed if isinstance(parsed, dict) else {})
    question = decision.get("question")
    question = question.strip() if isinstance(question, str) else ""
    items = list(by_id.values())
    chosen = []
    ambiguous = False
    for token in _id_tokens(decision):
        match = _match_token(token, items)
        if match == "ambiguous":
            ambiguous = True
        elif match is not None and match not in chosen:
            chosen.append(match)
    if len(chosen) == 1 and not ambiguous:
        return chosen[0]
    if ambiguous or len(chosen) > 1:
        return _clarify(question or "Which object do you mean?", items)
    if question:
        return _clarify(question, items)
    return _clarify(_seen_sentence(items), items)


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
