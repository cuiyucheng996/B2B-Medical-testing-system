"""B2B 辨证问诊 API。"""



from __future__ import annotations



import logging

from pathlib import Path

from typing import Any, Literal



from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from starlette.requests import Request



from configs.paths import ensure_b2b_src



ensure_b2b_src()



from algorithm.access_guard import EmptyTextError, TextTooLongError

from api.access_middleware import b2b_sessions_access_guard

from algorithm.step_gateway import StepGatewayError, prepare_step_update

from graph.session import get_b2b_state, invoke_b2b, new_thread_id

from graph.state import initial_b2b_state



logger = logging.getLogger(__name__)



WEB_DIR = Path(__file__).resolve().parent.parent / "web"





class SessionCreateRequest(BaseModel):

    user_id: str | None = None

    org_id: str | None = None





class StepRequest(BaseModel):

    message: str | None = None

    text: str | None = None

    selected_body_part: str | None = None

    selected_disease: str | None = None

    selected_symptoms: list[str] = Field(default_factory=list)

    org_id: str | None = None

    input_channel: Literal["text", "asr"] | None = None





def _jsonable(value: Any) -> Any:

    if value is None or isinstance(value, (str, int, float, bool)):

        return value

    if isinstance(value, dict):

        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):

        return [_jsonable(item) for item in value]

    content = getattr(value, "content", None)

    msg_type = getattr(value, "type", None)

    if content is not None and msg_type is not None:

        return {"role": str(msg_type), "content": str(content)}

    if hasattr(value, "model_dump"):

        try:

            return _jsonable(value.model_dump())

        except Exception:

            pass

    if hasattr(value, "isoformat"):

        try:

            return value.isoformat()

        except Exception:

            pass

    return str(value)





def _uniq_alignments(rows: list[Any]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            row.get("mention"),
            row.get("standard_name"),
            row.get("method"),
            row.get("node_label"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _compact_messages(values: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in values.get("messages") or []:
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("type") or "ai")
            content = str(item.get("content") or "")
        else:
            role = str(getattr(item, "type", None) or getattr(item, "role", "ai") or "ai")
            content = str(getattr(item, "content", "") or "")
        rows.append({"role": role, "content": content})
    return rows


def _compact_spell(values: dict[str, Any]) -> dict[str, Any] | None:
    detail = values.get("spell_correction_detail") or values.get("spell_correction")
    if not isinstance(detail, dict) or not detail:
        return None
    compact: dict[str, Any] = {
        "confidence": detail.get("confidence"),
        "explanation": detail.get("explanation"),
    }
    summary = values.get("spell_correction")
    if isinstance(summary, dict) and summary.get("source"):
        compact["source"] = summary.get("source")
    return compact


def _compact_channel_values(values: dict[str, Any]) -> dict[str, Any]:
    """测试台摘要：对话、纠错、意图、槽位；不展开证型树和 LangGraph 内部。"""
    slots = values.get("slots") if isinstance(values.get("slots"), dict) else {}
    syndromes = []
    for row in values.get("candidate_syndromes") or []:
        if isinstance(row, dict):
            syndromes.append(str(row.get("syndrome_name") or "").strip())
        else:
            syndromes.append(str(row))
    return {
        "messages": _compact_messages(values),
        "major_round": values.get("major_round"),
        "session_mode": values.get("session_mode"),
        "next_action": values.get("next_action"),
        "body_part": values.get("body_part"),
        "disease_name": values.get("disease_name"),
        "raw_query": values.get("raw_query") or "",
        "query": values.get("query") or "",
        "spell_correction": _compact_spell(values),
        "text_intent": values.get("text_intent"),
        "text_intent_confidence": values.get("text_intent_confidence"),
        "text_intent_reason": values.get("text_intent_reason"),
        "text_intent_source": values.get("text_intent_source"),
        "collected_symptoms": values.get("collected_symptoms") or [],
        "pending_current_main_hits": values.get("pending_current_main_hits") or [],
        "rollback_candidate_diseases": values.get("rollback_candidate_diseases") or [],
        "rollback_decision": values.get("rollback_decision"),
        "symptom_options": values.get("symptom_options") or [],
        "candidate_syndromes": [name for name in syndromes if name],
        "slots": {
            "schema": slots.get("schema") or [],
            "extracted": slots.get("extracted") or {},
            "mentions": slots.get("mentions") or {},
            "entities": slots.get("entities") or {},
            "body_part": slots.get("body_part"),
            "body_part_id": slots.get("body_part_id"),
            "disease_name": slots.get("disease_name"),
            "disease_id": slots.get("disease_id"),
            "symptoms": slots.get("symptoms") or [],
            "missing": slots.get("missing") or [],
            "filled": slots.get("filled"),
            "entry_nodes": slots.get("entry_nodes") or {},
            "alignments": _uniq_alignments(list(slots.get("alignments") or [])),
            "negation": slots.get("negation") or {},
            "fill_source": slots.get("fill_source") or "",
            "fill_model": slots.get("fill_model") or "",
            "fill_reason": slots.get("fill_reason") or "",
        },
    }


def _checkpoint_view(snapshot: Any, values: dict[str, Any]) -> dict[str, Any]:
    """LangGraph StateSnapshot 的可 JSON 视图，供测试台对照。"""
    config = getattr(snapshot, "config", None) or {}
    configurable = config.get("configurable") if isinstance(config, dict) else {}
    metadata = getattr(snapshot, "metadata", None) or {}
    return {
        "thread_id": (configurable or {}).get("thread_id"),
        "checkpoint_id": (configurable or {}).get("checkpoint_id"),
        "step": metadata.get("step") if isinstance(metadata, dict) else None,
        "next": list(getattr(snapshot, "next", None) or ()),
        "channel_values": _compact_channel_values(values if isinstance(values, dict) else {}),
    }





def _build_state_view(

    thread_id: str,

    values: dict[str, Any],

    snapshot: Any,

    *,

    gate: dict[str, Any] | None = None,

    run: Any = None,

) -> dict[str, Any]:

    paused = list(snapshot.next or ())

    view = {

        "thread_id": thread_id,

        "paused_nodes": paused,

        "major_round": values.get("major_round"),

        "session_mode": values.get("session_mode"),

        "next_action": values.get("next_action"),

        "body_part": values.get("body_part"),

        "disease_name": values.get("disease_name"),

        "body_part_options": values.get("body_part_options") or [],

        "disease_options": values.get("disease_options") or [],

        "symptom_options": values.get("symptom_options") or [],

        "collected_symptoms": values.get("collected_symptoms") or [],

        "candidate_syndromes": values.get("candidate_syndromes") or [],

        "matched_syndrome": values.get("matched_syndrome"),

        "final_syndromes": values.get("final_syndromes") or [],

        "auxiliary_round": values.get("auxiliary_round", 0),

        "input_channel": values.get("input_channel") or "text",

        "raw_query": values.get("raw_query") or "",

        "query": values.get("query") or "",

        "spell_correction": values.get("spell_correction"),

        "spell_correction_detail": values.get("spell_correction_detail"),

        "text_intent": values.get("text_intent"),

        "text_intent_confidence": values.get("text_intent_confidence"),

        "text_intent_reason": values.get("text_intent_reason"),

        "text_intent_source": values.get("text_intent_source"),

        "slots": values.get("slots"),

        "messages": [

            {"role": getattr(m, "type", "ai"), "content": str(getattr(m, "content", ""))}

            for m in values.get("messages") or []

        ],

        "values": values,

        "ok": True,

        "event": "done",

        "waiting_for_user": bool(paused),

        "error": None,

        "checkpoint": None,

    }

    try:

        view["checkpoint"] = _checkpoint_view(snapshot, values)

    except Exception as exc:

        logger.warning("serialize checkpoint failed: %s", exc)

        view["checkpoint"] = {"error": {"code": "checkpoint_serialize_failed", "message": str(exc)}}

    if run is not None:

        view["ok"] = bool(getattr(run, "ok", True))

        view["event"] = getattr(run, "event", "done")

        view["error"] = getattr(run, "error", None)

        if view["event"] == "interrupted":

            view["waiting_for_user"] = True

    if gate is not None:

        view["gate"] = gate

        view["ok"] = False

        view["event"] = "error"

        view["error"] = {

            "code": gate.get("category") or "gate",

            "message": gate.get("message") or "请求被拦截",

        }

    if view.get("error") and view["event"] == "error":

        err_text = str((view["error"] or {}).get("message") or "").strip()

        if err_text:

            view["messages"] = list(view["messages"]) + [

                {"role": "ai", "content": err_text}

            ]

    return view





def create_app() -> FastAPI:

    app = FastAPI(title="B2B Consult API", version="0.3.0")

    app.add_middleware(

        CORSMiddleware,

        allow_origins=["*"],

        allow_credentials=True,

        allow_methods=["*"],

        allow_headers=["*"],

        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],

    )



    app.middleware("http")(b2b_sessions_access_guard)



    @app.get("/api/meta")

    def get_meta() -> dict[str, Any]:

        from configs.b2b_api import get_b2b_api_config

        cfg = get_b2b_api_config()

        key_set = bool(str(cfg.get("api_key") or "").strip())

        return {

            "service": "b2b",

            "flow": "body->disease->main_symptom->auxiliary_syndrome",

            "auth": {

                "api_key_required": bool(cfg["api_key_required"]) or key_set,

                "header": "X-API-Key",

            },

            "rate_limit": {

                "enabled": bool(cfg["rate_limit_enabled"]),

                "max_requests": int(cfg["rate_limit_max_requests"]),

                "window_seconds": int(cfg["rate_limit_window_seconds"]),

            },

            "pipeline": [

                "access_guard",

                "input_channel",

                "text_guard",

                "intent_gate",

            ],

        }



    @app.post("/api/sessions")

    def create_session(req: SessionCreateRequest | None = None) -> dict[str, Any]:

        req = req or SessionCreateRequest()

        thread_id = new_thread_id()

        start = initial_b2b_state()

        if req.user_id:

            start["user_id"] = req.user_id.strip()

        if req.org_id:

            start["org_id"] = req.org_id.strip()

        run = invoke_b2b(start, thread_id=thread_id)

        snapshot = get_b2b_state(thread_id)

        values = dict(snapshot.values or run.values or {})

        return _build_state_view(thread_id, values, snapshot, run=run)



    @app.get("/api/sessions/{thread_id}")

    def get_session(thread_id: str) -> dict[str, Any]:

        snapshot = get_b2b_state(thread_id)

        if not snapshot.values:

            raise HTTPException(status_code=404, detail="会话不存在")

        return _build_state_view(thread_id, dict(snapshot.values), snapshot)



    @app.post("/api/sessions/{thread_id}/step")

    def step_session(thread_id: str, req: StepRequest, request: Request) -> dict[str, Any]:

        snapshot = get_b2b_state(thread_id)

        if not snapshot.values:

            raise HTTPException(status_code=404, detail="会话不存在")



        try:

            prepared = prepare_step_update(

                thread_id=thread_id,

                message=req.message,

                text=req.text,

                selected_body_part=req.selected_body_part,

                selected_disease=req.selected_disease,

                selected_symptoms=req.selected_symptoms,

                org_id=req.org_id,

                input_channel=req.input_channel,

                request=request,

            )

        except TextTooLongError as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except (StepGatewayError, EmptyTextError) as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



        if prepared.blocked:

            gate = {

                "blocked": True,

                "category": prepared.block_category,

                "message": prepared.gate_message,

                "intent": (

                    {

                        "category": prepared.intent.category,

                        "reason": prepared.intent.reason,

                    }

                    if prepared.intent is not None

                    else None

                ),

            }

            if prepared.block_category == "sensitive":

                raise HTTPException(status_code=400, detail=prepared.gate_message or "内容含敏感信息")

            snapshot = get_b2b_state(thread_id)

            return _build_state_view(

                thread_id,

                dict(snapshot.values),

                snapshot,

                gate=gate,

            )



        run = invoke_b2b(prepared.update, thread_id=thread_id)

        snapshot = get_b2b_state(thread_id)

        values = dict(snapshot.values or run.values or {})

        return _build_state_view(thread_id, values, snapshot, run=run)



    if WEB_DIR.is_dir():

        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="b2b-static")



        @app.get("/")

        def index() -> FileResponse:

            return FileResponse(

                WEB_DIR / "index.html",

                headers={"Cache-Control": "no-store, max-age=0"},

            )



    return app





app = create_app()

