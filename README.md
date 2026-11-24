# St. Jude Outpatient — Voice Scheduling

A small, in-memory clinic front desk with a live appointment board and a browser-based voice receptionist. The call runs over WebRTC: Pipecat streams browser audio through ElevenLabs speech recognition, Silero VAD handles turn detection and interruptions, Gemini `gemini-2.5-flash` decides when to use the availability, booking, or cancellation tools, and ElevenLabs speaks the response. A transcript processor mirrors the caller and receptionist turns to the dashboard.

Without provider keys, **Call Reception** starts a clearly labeled local demo using browser speech recognition and speech synthesis (or typed messages if browser speech input is unavailable). The demo exercises the same in-memory appointment state and confirmation flow; it does not call Gemini or ElevenLabs.

## Run locally

Use Python 3.10 or newer. Create `.env` from `.env.example` and add your Gemini and ElevenLabs API keys. Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in a browser with microphone support. Clinic data is held in memory and resets when the server restarts. For provider-backed calls, allow microphone access and configure both API keys. The local demo needs no API keys.

## Call flow

The receptionist checks doctor availability, reads back the caller's phone number one digit at a time, and waits for an explicit yes before either booking or cancelling. Tool handlers independently verify the spoken phone number, confirmation, and requested slot before changing the clinic state.
