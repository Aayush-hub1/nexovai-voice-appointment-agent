import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinic-agent")

SLOTS = ["09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM", "11:00 AM", "01:30 PM", "02:00 PM", "03:30 PM"]
TODAY = date.today()
DOCTORS: list[dict[str, Any]] = [
    {
        "id": "dr-olivia-chen",
        "name": "Dr. Olivia Chen",
        "specialty": "Family Medicine",
        "room": "Room 204",
        "available_slots": [
            {"id": f"chen-{TODAY + timedelta(days=offset)}-{time}", "date": (TODAY + timedelta(days=offset)).isoformat(), "time": time}
            for offset, time in [(0, "09:30 AM"), (0, "11:00 AM"), (1, "10:00 AM"), (2, "02:00 PM")]
        ],
    },
    {
        "id": "dr-marcus-reed",
        "name": "Dr. Marcus Reed",
        "specialty": "Cardiology",
        "room": "Room 118",
        "available_slots": [
            {"id": f"reed-{TODAY + timedelta(days=offset)}-{time}", "date": (TODAY + timedelta(days=offset)).isoformat(), "time": time}
            for offset, time in [(0, "10:00 AM"), (1, "09:00 AM"), (1, "01:30 PM"), (3, "03:30 PM")]
        ],
    },
    {
        "id": "dr-amara-williams",
        "name": "Dr. Amara Williams",
        "specialty": "Dermatology",
        "room": "Room 306",
        "available_slots": [
            {"id": f"williams-{TODAY + timedelta(days=offset)}-{time}", "date": (TODAY + timedelta(days=offset)).isoformat(), "time": time}
            for offset, time in [(0, "01:30 PM"), (1, "10:30 AM"), (2, "09:30 AM"), (4, "02:00 PM")]
        ],
    },
]
INITIAL_DOCTOR_SLOTS = {
    doctor["id"]: [dict(slot) for slot in doctor["available_slots"]]
    for doctor in DOCTORS
}
APPOINTMENTS: list[dict[str, Any]] = []
TRANSCRIPTS: list[dict[str, Any]] = []
LATEST_EVENT: dict[str, Any] | None = None
SESSIONS: dict[str, dict[str, Any]] = {}
DEMO_SESSIONS: dict[str, dict[str, Any]] = {}
SESSION_TASKS: set[asyncio.Task[Any]] = set()
STATE_LOCK = asyncio.Lock()
WEBRTC_HANDLER: Any = None

SYSTEM_PROMPT = """You are the calm, direct receptionist at St. Jude Outpatient. Keep every response to one or two short sentences. Use no markdown, bullets, or robotic filler phrases. Identify callers only by the phone number they speak; there is no login or OTP. Treat common speech-recognition substitutions in phone numbers as digits, including "to" or "too" for two and "for" for four. When the caller gives a 10-digit phone number, normalize it to digits, read those 10 digits back once for confirmation, and immediately call book_appointment or cancel_appointment as soon as the caller says yes to that exact confirmation. Do not ask them to repeat a clear number.

For booking or cancellation, NEVER invoke the corresponding tool until you have read the caller's phone number back digit by digit, stated the selected appointment/booking, and asked them to confirm with yes. Only invoke the tool after an explicit yes to that exact confirmation. Ask one question at a time. For a booking, first check availability, then confirm the specific doctor, date, time and phone number. Never claim an appointment is booked or cancelled unless its tool succeeds. Do not provide medical advice."""


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    tasks = list(SESSION_TASKS)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    if WEBRTC_HANDLER is not None:
        await WEBRTC_HANDLER.close()


app = FastAPI(title="St. Jude Outpatient Voice Scheduling", lifespan=lifespan)


class OfferPayload(BaseModel):
    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool = False
    request_data: dict[str, Any] | None = None


class IceCandidatePayload(BaseModel):
    candidate: str
    sdp_mid: str
    sdp_mline_index: int


class IcePatchPayload(BaseModel):
    pc_id: str
    candidates: list[IceCandidatePayload]


class DemoTurn(BaseModel):
    session_id: str
    message: str


class DemoSession(BaseModel):
    session_id: str


NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "to": "2", "too": "2", "four": "4", "for": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _phone_digits(phone: str) -> str:
    tokens = re.findall(r"\d+|[a-z]+", phone.casefold())
    return "".join(
        NUMBER_WORDS.get(token, token if token.isdigit() else "")
        for token in tokens
    )


def _spoken_phone(text: str) -> str | None:
    tokens = re.findall(r"\d+|[a-z]+", text.casefold())
    number_runs: list[str] = []
    current = ""
    for token in tokens:
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
        elif token.isdigit():
            current += token
        else:
            if len(current) >= 10:
                number_runs.append(current)
            current = ""
    if len(current) >= 10:
        number_runs.append(current)
    if not number_runs:
        return None
    return number_runs[0][-10:]


def _is_affirmative(text: str) -> bool:
    normalized = re.sub(r"[^a-z ]", "", text.casefold()).strip()
    return bool(re.fullmatch(r"(yes|yeah|yep|correct|right|i confirm|that is correct|thats correct)( please)?", normalized))


def _is_negative(text: str) -> bool:
    normalized = re.sub(r"[^a-z ]", "", text.casefold()).strip()
    return bool(re.fullmatch(r"(no|nope|not now|i do not confirm|dont)( please)?", normalized))


def _spoken_digits(phone: str) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return ", ".join(words[int(digit)] for digit in phone[-10:])


def _slot_sort_key(slot: dict[str, str]) -> datetime:
    return datetime.strptime(f"{slot['date']} {slot['time']}", "%Y-%m-%d %I:%M %p")


def _record_transcript(session_id: str, role: str, text: str) -> None:
    normalized_text = " ".join(text.split()).casefold()
    previous = next(
        (entry for entry in reversed(TRANSCRIPTS) if entry["session_id"] == session_id),
        None,
    )
    if previous and previous["role"] == role and " ".join(previous["text"].split()).casefold() == normalized_text:
        return
    TRANSCRIPTS.append({
        "id": uuid.uuid4().hex,
        "session_id": session_id,
        "role": role,
        "text": text.strip(),
        "at": datetime.now().isoformat(timespec="seconds"),
    })
    if len(TRANSCRIPTS) > 400:
        del TRANSCRIPTS[:-300]


def _requested_doctor(text: str) -> dict[str, Any] | None:
    normalized = text.casefold()
    aliases = {
        "Dr. Olivia Chen": ("olivia", "chen", "family", "family medicine", "general medicine"),
        "Dr. Marcus Reed": ("marcus", "reed", "cardiology", "cardiologist"),
        "Dr. Amara Williams": ("amara", "williams", "dermatology", "dermatologist"),
    }
    return next(
        (doctor for doctor in DOCTORS if any(alias in normalized for alias in aliases[doctor["name"]])),
        None,
    )


def _demo_confirmation_prompt(pending: dict[str, Any]) -> str:
    digits = _spoken_digits(pending["phone"])
    if pending["action"] == "booking":
        doctor = pending["doctor"]
        slot = pending["slot"]
        return (
            f"I can book {doctor['name']} on {slot['date']} at {slot['time']} under "
            f"{digits}. Should I book it? Please say yes."
        )
    appointment = pending["appointment"]
    return (
        f"I have {appointment['doctor']} on {appointment['date']} at {appointment['time']} "
        f"under {digits}. Should I cancel it? Please say yes."
    )


def _find_doctor(name: str | None = None, specialty: str | None = None) -> list[dict[str, Any]]:
    name_query = (name or "").strip().casefold()
    specialty_query = (specialty or "").strip().casefold()
    return [
        doctor for doctor in DOCTORS
        if (not name_query or name_query in doctor["name"].casefold())
        and (not specialty_query or specialty_query in doctor["specialty"].casefold())
    ]


def check_doctor_availability(doctor_name: str | None = None, specialty: str | None = None) -> dict[str, Any]:
    matches = _find_doctor(doctor_name, specialty)
    return {
        "doctors": [
            {
                "name": doctor["name"],
                "specialty": doctor["specialty"],
                "room": doctor["room"],
                "available_slots": list(doctor["available_slots"]),
            }
            for doctor in matches
        ]
    }


async def book_appointment(
    caller_phone: str,
    doctor_name: str,
    slot_id: str,
    confirmation: str,
) -> dict[str, Any]:
    global LATEST_EVENT
    digits = _phone_digits(caller_phone)
    if len(digits) < 10:
        return {"error": "A valid phone number with at least 10 digits is required."}
    if confirmation.strip().casefold() not in {"yes", "yes, i confirm", "confirmed"}:
        return {"error": "The caller must explicitly confirm before booking."}
    async with STATE_LOCK:
        doctor = next((item for item in DOCTORS if item["name"].casefold() == doctor_name.casefold()), None)
        if not doctor:
            return {"error": "That doctor could not be found."}
        slot = next((item for item in doctor["available_slots"] if item["id"] == slot_id), None)
        if not slot:
            return {"error": "That time is no longer available. Check availability again."}
        doctor["available_slots"].remove(slot)
        booking = {
            "ref": f"SJ-{uuid.uuid4().hex[:6].upper()}",
            "caller": digits,
            "doctor": doctor["name"],
            "specialty": doctor["specialty"],
            "room": doctor["room"],
            "date": slot["date"],
            "time": slot["time"],
            "status": "Confirmed",
        }
        APPOINTMENTS.insert(0, booking)
        LATEST_EVENT = {
            "id": uuid.uuid4().hex,
            "type": "booking",
            "appointment": booking,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    return {"success": True, "appointment": booking}


async def cancel_appointment(caller_phone: str, confirmation: str) -> dict[str, Any]:
    global LATEST_EVENT
    digits = _phone_digits(caller_phone)
    if len(digits) < 10:
        return {"error": "A valid phone number with at least 10 digits is required."}
    if confirmation.strip().casefold() not in {"yes", "yes, i confirm", "confirmed"}:
        return {"error": "The caller must explicitly confirm before cancelling."}
    async with STATE_LOCK:
        appointment_index = next(
            (i for i, appointment in enumerate(APPOINTMENTS)
             if _phone_digits(appointment["caller"]) == digits and appointment["status"] == "Confirmed"),
            None,
        )
        if appointment_index is None:
            return {"error": "No confirmed appointment was found for that phone number."}
        appointment = APPOINTMENTS.pop(appointment_index)
        doctor = next(item for item in DOCTORS if item["name"] == appointment["doctor"])
        doctor["available_slots"].append({
            "id": f"restored-{appointment['ref']}",
            "date": appointment["date"],
            "time": appointment["time"],
        })
        doctor["available_slots"].sort(key=_slot_sort_key)
        cancelled = {**appointment, "status": "Cancelled"}
        LATEST_EVENT = {
            "id": uuid.uuid4().hex,
            "type": "cancellation",
            "appointment": cancelled,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    return {"success": True, "appointment": cancelled}


def get_state() -> dict[str, Any]:
    return {
        "doctors": DOCTORS,
        "open_slots": sum(len(doctor["available_slots"]) for doctor in DOCTORS),
        "appointments": APPOINTMENTS,
        "transcripts": TRANSCRIPTS[-80:],
        "latest_event": LATEST_EVENT,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    return get_state()


@app.post("/api/reset")
async def reset_state() -> dict[str, Any]:
    global LATEST_EVENT
    async with STATE_LOCK:
        APPOINTMENTS.clear()
        TRANSCRIPTS.clear()
        LATEST_EVENT = None
        for doctor in DOCTORS:
            doctor["available_slots"] = [
                dict(slot) for slot in INITIAL_DOCTOR_SLOTS[doctor["id"]]
            ]
    return get_state()


@app.get("/api/config")
async def api_config() -> dict[str, bool]:
    return {
        "voice_enabled": bool(os.getenv("GEMINI_API_KEY") and os.getenv("ELEVENLABS_API_KEY"))
    }


@app.post("/api/demo/start")
async def demo_start() -> dict[str, str]:
    session_id = uuid.uuid4().hex
    DEMO_SESSIONS[session_id] = {"phone": None, "requested_intent": None, "doctor": None, "pending": None}
    reply = "Hello, this is St. Jude Outpatient. How can I help with your appointment?"
    _record_transcript(session_id, "assistant", reply)
    return {"session_id": session_id, "reply": reply}


@app.post("/api/demo/end")
async def demo_end(payload: DemoSession) -> dict[str, bool]:
    DEMO_SESSIONS.pop(payload.session_id, None)
    return {"ended": True}


@app.post("/api/demo/turn")
async def demo_turn(payload: DemoTurn) -> dict[str, Any]:
    session = DEMO_SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(404, "This local demo call has ended.")

    text = payload.message.strip()
    if not text:
        raise HTTPException(400, "Say or type a message to continue.")
    _record_transcript(payload.session_id, "user", text)
    pending = session.get("pending")

    if pending:
        if _is_affirmative(text):
            session["pending"] = None
            if pending["action"] == "booking":
                result = await book_appointment(
                    caller_phone=pending["phone"],
                    doctor_name=pending["doctor"]["name"],
                    slot_id=pending["slot"]["id"],
                    confirmation="yes",
                )
            else:
                result = await cancel_appointment(
                    caller_phone=pending["phone"],
                    confirmation="yes",
                )
            if result.get("success"):
                appointment = result["appointment"]
                reply = (
                    f"Your appointment is booked with {appointment['doctor']} on {appointment['date']} "
                    f"at {appointment['time']}. Your reference is {appointment['ref']}."
                    if pending["action"] == "booking"
                    else f"Your appointment {appointment['ref']} has been cancelled."
                )
            else:
                reply = result.get("error", "I could not complete that change. Please try again.")
            _record_transcript(payload.session_id, "assistant", reply)
            return {"reply": reply}
        if _is_negative(text):
            session["pending"] = None
            reply = "Okay. I have not made any changes."
            _record_transcript(payload.session_id, "assistant", reply)
            return {"reply": reply}
        corrected_phone = _spoken_phone(text)
        if corrected_phone:
            session["phone"] = corrected_phone
            pending["phone"] = corrected_phone
        reply = _demo_confirmation_prompt(pending)
        _record_transcript(payload.session_id, "assistant", reply)
        return {"reply": reply}

    phone = _spoken_phone(text)
    if phone:
        session["phone"] = phone

    lowered = text.casefold()
    if re.search(r"\b(cancel|cancellation)\b", lowered):
        session["requested_intent"] = "cancellation"
    elif re.search(r"\b(book|booking|schedule|appointment)\b", lowered):
        session["requested_intent"] = "booking"
        requested_doctor = _requested_doctor(text)
        if requested_doctor:
            session["doctor"] = requested_doctor["name"]

    intent = session.get("requested_intent")
    if intent == "cancellation":
        if not session.get("phone"):
            reply = "What phone number was the appointment booked under?"
        else:
            match = next(
                (appointment for appointment in APPOINTMENTS
                 if _phone_digits(appointment["caller"]) == session["phone"]
                 and appointment["status"] == "Confirmed"),
                None,
            )
            if not match:
                reply = "I could not find a confirmed appointment for that phone number."
                session["requested_intent"] = None
            else:
                session["pending"] = {
                    "action": "cancellation",
                    "phone": session["phone"],
                    "appointment": match,
                }
                reply = _demo_confirmation_prompt(session["pending"])
                session["requested_intent"] = None
    elif intent == "booking":
        if not session.get("phone"):
            reply = "What phone number should I use for the appointment?"
        else:
            doctor = next(
                (item for item in DOCTORS if item["name"] == session.get("doctor") and item["available_slots"]),
                None,
            )
            if not doctor:
                matches = _find_doctor()
                doctor = next((item for item in matches if item["available_slots"]), None)
            if not doctor:
                reply = "I could not find an open appointment time right now."
                session["requested_intent"] = None
            else:
                slot = doctor["available_slots"][0]
                session["pending"] = {
                    "action": "booking",
                    "phone": session["phone"],
                    "doctor": doctor,
                    "slot": slot,
                }
                reply = _demo_confirmation_prompt(session["pending"])
                session["requested_intent"] = None
    elif any(word in lowered for word in ("available", "availability", "opening", "open slot", "doctors")):
        doctor = _requested_doctor(text)
        results = check_doctor_availability(
            doctor_name=doctor["name"] if doctor else None,
        )["doctors"]
        openings = [
            (item, slot)
            for item in results
            for slot in item["available_slots"]
        ][:3]
        if openings:
            first_doctor, first_slot = openings[0]
            remaining = len(openings) - 1
            reply = (
                f"{first_doctor['name']} has an opening on {first_slot['date']} at {first_slot['time']}."
                + (f" I found {remaining} more nearby times." if remaining else "")
            )
        else:
            reply = "I could not find an open time for that doctor."
    elif phone:
        reply = "Thanks. Would you like me to check availability, book a visit, or cancel an appointment?"
    else:
        reply = "I can check availability, book a visit, or cancel an appointment. What would you like to do?"

    _record_transcript(payload.session_id, "assistant", reply)
    return {"reply": reply}


@app.post("/api/offer")
async def offer(payload: OfferPayload) -> dict[str, Any]:
    global WEBRTC_HANDLER
    missing = [key for key in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY") if not os.getenv(key)]
    if missing:
        raise HTTPException(503, f"Configure {', '.join(missing)} in .env before starting a call.")
    try:
        from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest, SmallWebRTCRequestHandler
    except ImportError as exc:
        logger.exception("Pipecat WebRTC transport import failed")
        raise HTTPException(500, "Pipecat WebRTC dependencies are not installed correctly.") from exc

    if WEBRTC_HANDLER is None:
        WEBRTC_HANDLER = SmallWebRTCRequestHandler()
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "caller_phone": None,
        "pending_confirmation": None,
        "confirmed_action": None,
    }
    SESSIONS[session_id] = session
    try:
        async def start_pipeline(connection: Any) -> None:
            @connection.event_handler("closed")
            async def close_session(_: Any) -> None:
                SESSIONS.pop(session_id, None)

            task = asyncio.create_task(run_pipeline(connection, session))
            SESSION_TASKS.add(task)

            def pipeline_finished(done: asyncio.Task[Any]) -> None:
                SESSION_TASKS.discard(done)
                if done.cancelled():
                    return
                error = done.exception()
                if error:
                    logger.error(
                        "Pipecat voice pipeline failed: %s",
                        error,
                        exc_info=(type(error), error, error.__traceback__),
                    )

            task.add_done_callback(pipeline_finished)

        answer = await WEBRTC_HANDLER.handle_web_request(
            SmallWebRTCRequest(
                sdp=payload.sdp,
                type=payload.type,
                pc_id=payload.pc_id,
                restart_pc=payload.restart_pc,
                request_data=payload.request_data,
            ),
            start_pipeline,
        )
        if answer is None:
            raise RuntimeError("Pipecat did not create an SDP answer.")
        return {**answer, "session_id": session_id}
    except Exception as exc:
        SESSIONS.pop(session_id, None)
        logger.exception("Could not initialize WebRTC call")
        raise HTTPException(500, "Could not establish the voice connection.") from exc


@app.patch("/api/offer")
async def patch_offer(payload: IcePatchPayload) -> dict[str, str]:
    if WEBRTC_HANDLER is None:
        raise HTTPException(404, "WebRTC peer connection was not found.")
    from pipecat.transports.smallwebrtc.request_handler import IceCandidate, SmallWebRTCPatchRequest

    await WEBRTC_HANDLER.handle_patch_request(
        SmallWebRTCPatchRequest(
            pc_id=payload.pc_id,
            candidates=[IceCandidate(**candidate.model_dump()) for candidate in payload.candidates],
        )
    )
    return {"status": "success"}


async def run_pipeline(connection: Any, session: dict[str, Any]) -> None:
    import aiohttp
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
    from pipecat.frames.frames import (
        Frame,
        InterimTranscriptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TTSTextFrame,
        TranscriptionFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.google.llm import GoogleLLMService
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    class TranscriptProcessor(FrameProcessor):
        def __init__(self, role: str):
            super().__init__()
            self.role = role
            self.assistant_text = ""

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if self.role == "caller" and isinstance(frame, TranscriptionFrame) and not isinstance(frame, InterimTranscriptionFrame):
                text = frame.text.strip()
                if text:
                    _record_transcript(session["id"], "user", text)
                    spoken_phone = _spoken_phone(text)
                    if spoken_phone:
                        session["caller_phone"] = spoken_phone
                    if _is_affirmative(text) and session.get("pending_confirmation"):
                        session["confirmed_action"] = session.pop("pending_confirmation")
                    elif session.get("pending_confirmation"):
                        session["pending_confirmation"] = None
                        session["confirmed_action"] = None

            if self.role == "reception":
                if isinstance(frame, LLMFullResponseStartFrame):
                    self.assistant_text = ""
                elif isinstance(frame, TTSTextFrame):
                    self.assistant_text += frame.text + " "
                elif isinstance(frame, LLMFullResponseEndFrame):
                    text = self.assistant_text.strip()
                    if text:
                        _record_transcript(session["id"], "assistant", text)
                        lowered = text.casefold()
                        phone_readback = _spoken_phone(text)
                        action = "booking" if re.search(r"\b(book|booking|reserve|reservation)\b", lowered) else (
                            "cancellation" if re.search(r"\b(cancel|cancellation)\b", lowered) else None
                        )
                        asks_confirmation = (
                            "yes" in lowered
                            or "confirm" in lowered
                            or "should i" in lowered
                            or "would you like me to" in lowered
                        )
                        if (
                            action
                            and asks_confirmation
                            and session.get("caller_phone")
                            and phone_readback == session["caller_phone"]
                        ):
                            session["pending_confirmation"] = action
                    self.assistant_text = ""
                    if len(TRANSCRIPTS) > 400:
                        del TRANSCRIPTS[:-300]
            await self.push_frame(frame, direction)

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    context = LLMContext(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "A caller has just connected. Greet them briefly and ask what appointment they need."},
        ],
        tools=[
            FunctionSchema(
                name="check_doctor_availability",
                description="Check open appointment slots by optional doctor name or specialty.",
                properties={
                    "doctor_name": {"type": "string", "description": "Doctor's name, if specified"},
                    "specialty": {"type": "string", "description": "Medical specialty, if specified"},
                },
                required=[],
            ),
            FunctionSchema(
                name="book_appointment",
                description="Book one available slot only after the caller confirmed a digit-by-digit phone number and this exact appointment.",
                properties={
                    "caller_phone": {"type": "string"},
                    "doctor_name": {"type": "string"},
                    "slot_id": {"type": "string"},
                    "confirmation": {"type": "string"},
                },
                required=["caller_phone", "doctor_name", "slot_id", "confirmation"],
            ),
            FunctionSchema(
                name="cancel_appointment",
                description="Cancel an appointment only after the caller confirmed a digit-by-digit phone number and cancellation.",
                properties={
                    "caller_phone": {"type": "string"},
                    "confirmation": {"type": "string"},
                },
                required=["caller_phone", "confirmation"],
            ),
        ],
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=True)],
            ),
        ),
    )
    llm = GoogleLLMService(
        api_key=os.environ["GEMINI_API_KEY"],
        settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
    )

    async def availability_tool(params: Any) -> None:
        result = check_doctor_availability(
            params.arguments.get("doctor_name"),
            params.arguments.get("specialty"),
        )
        await params.result_callback(result)

    async def book_tool(params: Any) -> None:
        args = params.arguments
        if (
            session.get("caller_phone") != _phone_digits(args.get("caller_phone", ""))[-10:]
            or session.get("confirmed_action") != "booking"
            or args.get("confirmation", "").strip().casefold() not in {"yes", "yes, i confirm", "confirmed"}
        ):
            await params.result_callback({"error": "Caller number must match the phone number spoken on this call."})
            return
        result = await book_appointment(**args)
        if result.get("success"):
            session["confirmed_action"] = None
        await params.result_callback(result)

    async def cancel_tool(params: Any) -> None:
        args = params.arguments
        if (
            session.get("caller_phone") != _phone_digits(args.get("caller_phone", ""))[-10:]
            or session.get("confirmed_action") != "cancellation"
            or args.get("confirmation", "").strip().casefold() not in {"yes", "yes, i confirm", "confirmed"}
        ):
            await params.result_callback({"error": "Caller number must match the phone number spoken on this call."})
            return
        result = await cancel_appointment(**args)
        if result.get("success"):
            session["confirmed_action"] = None
        await params.result_callback(result)

    llm.register_function("check_doctor_availability", availability_tool)
    llm.register_function("book_appointment", book_tool)
    llm.register_function("cancel_appointment", cancel_tool)

    async with aiohttp.ClientSession() as aiohttp_session:
        stt = ElevenLabsSTTService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            aiohttp_session=aiohttp_session,
        )
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
            ),
            aiohttp_session=aiohttp_session,
        )
        pipeline = Pipeline([
            transport.input(),
            stt,
            TranscriptProcessor("caller"),
            context_aggregator.user(),
            llm,
            tts,
            TranscriptProcessor("reception"),
            transport.output(),
            context_aggregator.assistant(),
        ])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        from pipecat.frames.frames import LLMRunFrame

        await task.queue_frames([LLMRunFrame()])
        await PipelineRunner().run(task)
