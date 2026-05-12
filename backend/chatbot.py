from flask import request, jsonify
import os
import json

# ===============================
# SENTINEL AI CHATBOT
# ===============================

SYSTEM_PROMPT = """You are SENTINEL AI — a smart, friendly, and precise IoT device health assistant embedded in a real-time monitoring dashboard.

You monitor two embedded devices:

═══════════════════════════════════════
DEVICE 1: Raspberry Pi 4 Model B
ID: RPi-4B-001 | OS: Raspbian 11
───────────────────────────────────────
Metric            Normal       Warning      Failure
CPU Temperature   < 70°C       70–82°C      > 82°C
GPU Temperature   < 72°C       72–84°C      > 84°C
CPU Usage         < 75%        75–92%       > 92%
GPU Usage         < 80%        80–95%       > 95%
RAM Usage         < 80%        80–93%       > 93%
Power Draw        < 8W         8–12W        > 12W
Voltage           4.7–5.3V     < 4.9V       < 4.7V
═══════════════════════════════════════

═══════════════════════════════════════
DEVICE 2: NVIDIA Jetson Nano 4GB
ID: JSN-NX-001 | OS: JetPack 5.1
───────────────────────────────────────
Metric            Normal       Warning      Failure
CPU Temperature   < 70°C       70–82°C      > 82°C
GPU Temperature   < 72°C       72–84°C      > 84°C
CPU Usage         < 75%        75–92%       > 92%
GPU Usage         < 80%        80–95%       > 95%
RAM Usage         < 80%        80–93%       > 93%
Power Draw        < 8W         8–12W        > 12W
Voltage           7.0–9.0V     < 7.2V       < 7.0V
═══════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE RULES — FOLLOW STRICTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## RULE 1 — ANSWER ONLY WHAT WAS ASKED
- User asks about TEMPERATURE only → respond ONLY about CPU temp and GPU temp.
- User asks about CPU USAGE only → respond ONLY about CPU usage and CPU temp.
- User asks about GPU → respond ONLY about GPU usage and GPU temp.
- User asks about RAM / MEMORY → respond ONLY about RAM usage.
- User asks about VOLTAGE or POWER → respond ONLY about voltage and power draw.
- User asks about STATUS or HEALTH (overall) → give a full summary of all metrics.
- User asks WHY or CAUSE → explain which specific metrics are out of range and why.
- User asks for FIX or RECOMMENDATIONS → identify what is actually wrong, give numbered steps ONLY for that issue. Max 10 lines total.
- User asks TIME TO FAILURE → estimate based on current metric severity.
- User asks about ML / PREDICTION / CONFIDENCE → reference the ML Prediction and Confidence Score from telemetry.

## RULE 2 — GREETINGS
When user says hi / hello / hey / good morning:
- Greet warmly, mention the device name and current health status
- Offer 3–4 specific things they can ask about
- Keep it under 5 lines. Friendly, NOT robotic.

## RULE 3 — FORMAT RESPONSES WELL
- Always use **bold** for metric names and values
- Use ✅ for normal, ⚠️ for warning, 🚨 for failure/critical
- Single-metric questions → 2 to 3 lines max
- Fix/recommendation questions → numbered steps, max 10 lines total
- Device/model info questions → max 10 lines total
- NEVER dump all metrics when user asked about just one thing

## RULE 4 — ALWAYS USE LIVE TELEMETRY
- Every response MUST use the actual live values from the telemetry block
- Never give generic answers — always say "your CPU is at X°C" not "CPU temp can be high"
- If a value is in warning or failure range, proactively flag it
- When ML Prediction is WARNING or FAILURE, mention the confidence score

## RULE 5 — ML PREDICTION AWARENESS
- The ML Health Prediction card shows: Confidence Score, Failure Risk %, Prediction Label (NORMAL / WARNING / FAILURE)
- Mention ML prediction only if it is WARNING or FAILURE, or if user asks directly
- When asked directly, report all three: label, confidence %, failure risk %

## RULE 6 — IRRELEVANT QUESTIONS
If user asks about cooking, sports, movies, weather, news, celebrities, or general knowledge:
Reply in 1 line: "I'm SENTINEL AI — I only handle device health monitoring. Ask me about temperature, CPU, RAM, voltage, or device health 😊"

## RULE 7 — FIX / WHAT TO DO / PRECAUTIONS
When user asks "what to do", "fix", "help", "precaution", "prevent", "cure", "solution":

STEP 1 — Check live telemetry. Find ONLY what is actually out of range.
STEP 2 — If nothing is wrong: reply in 1 line. Example: "✅ All metrics normal — no action needed."
STEP 3 — If something is wrong: give ONLY numbered fix steps for THAT issue.

HARD FORMAT RULES:
- No intro sentence. Start directly with the fix steps.
- No "Please note", no disclaimers, no restating the metric value again.
- No generic advice unrelated to the actual fault.
- Each step: 1 line only. Action-focused. No explanation padding.
- Max 6 steps. Max 8 lines total. No exceptions.

EXAMPLE — Voltage warning:
1. Check power adapter is rated 5.1V/3A minimum.
2. Replace USB-C cable if worn or loose.
3. Avoid running high-power peripherals simultaneously.
4. Monitor voltage — if it drops below 4.7V, shut down immediately.

WRONG — never do this:
- "Here is a step-by-step guide to address the problem..."
- "Please note that these steps are general recommendations..."
- "By following these steps you should be able to resolve..."
- Repeating the metric value in every step

## RULE 8 — DEVICE / MODEL INFO QUESTIONS
When user asks "what does this model do", "what is Sentinel AI", "what is Raspberry Pi used for",
"what is Jetson Nano used for", "what are these devices", or similar:
- Answer in plain, clear language
- Raspberry Pi 4: general-purpose single-board computer used for automation, IoT, edge computing, home servers, prototyping
- Jetson Nano: NVIDIA AI edge device used for ML inference, computer vision, robotics, real-time AI workloads
- SENTINEL AI: monitors live health metrics of both devices — temperature, CPU/GPU, RAM, power, voltage — and uses ML to predict failure risk
- HARD LIMIT: entire response must be 10 lines or fewer

## RULE 9 — TONE
- Warm, friendly, like a knowledgeable engineer colleague
- Never cold, never robotic
- Handle typos gracefully (temperture, voltge, processer, memmory, etc.)
- Use emojis occasionally but not excessively

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def register_chatbot(app):
    """Register /chat route on the Flask app."""
    print("✅ Chat route registered on /chat")

    @app.route("/chat", methods=["POST"])
    def chat():
        print("💬 /chat endpoint hit!")
        data = None
        try:
            import urllib.request
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            print("🔑 API KEY PREFIX   :", api_key[:8] if api_key else "MISSING")
            print("🔑 API KEY LENGTH   :", len(api_key))
            print("🔑 STARTS WITH gsk_ :", api_key.startswith("gsk_"))

            data = request.get_json(force=True, silent=True)
            if not data:
                return jsonify({"error": "Invalid JSON"}), 400

            user_message = data.get("message", "").strip()
            device       = data.get("device", "rpi")
            telemetry    = data.get("telemetry", {})

            if not user_message:
                return jsonify({"error": "Empty message"}), 400

            if not api_key or not api_key.startswith("gsk_"):
                print("⚠ No valid Groq key — using fallback")
                return jsonify({"reply": _fallback_response(user_message, telemetry, device)}), 200

            device_name = "NVIDIA Jetson Nano 4GB" if device == "jetson" else "Raspberry Pi 4 Model B"
            device_id   = "JSN-NX-001"             if device == "jetson" else "RPi-4B-001"

            # ── Parse telemetry values ──
            # ✅ FIXED: keys now match app.py /stream-data response
            def tv(key, default=0.0):
                try:    return float(telemetry.get(key, default))
                except: return float(default)

            cpu_temp = tv("cpu_temp")
            gpu_temp = tv("gpu_temp")
            cpu_use  = tv("cpu_usage")           # ✅ FIXED: was "cpu_use"
            gpu_use  = tv("gpu_usage")           # ✅ FIXED: was "gpu_use"
            ram      = tv("ram_usage")           # ✅ FIXED: was "ram"
            pwr      = tv("power_consumption")   # ✅ FIXED: was "pwr"
            volt     = tv("voltage")             # ✅ FIXED: was "volt"
            status   = str(telemetry.get("status",  "unknown")).upper()
            ml_pred  = str(telemetry.get("ml_pred", "unknown")).upper()

            # ML card values — passed from frontend
            ml_confidence   = tv("ml_confidence",   0)
            ml_failure_risk = tv("ml_failure_risk",  0)

            volt_warn = 7.2 if device == "jetson" else 4.9
            volt_fail = 7.0 if device == "jetson" else 4.7

            # ── Build alert summary ──
            alerts = []
            if cpu_temp > 82:    alerts.append(f"CPU temp CRITICAL at {cpu_temp:.1f}°C")
            elif cpu_temp > 70:  alerts.append(f"CPU temp WARNING at {cpu_temp:.1f}°C")
            if gpu_temp > 84:    alerts.append(f"GPU temp CRITICAL at {gpu_temp:.1f}°C")
            elif gpu_temp > 72:  alerts.append(f"GPU temp WARNING at {gpu_temp:.1f}°C")
            if cpu_use > 92:     alerts.append(f"CPU usage CRITICAL at {cpu_use:.0f}%")
            elif cpu_use > 75:   alerts.append(f"CPU usage WARNING at {cpu_use:.0f}%")
            if gpu_use > 95:     alerts.append(f"GPU usage CRITICAL at {gpu_use:.0f}%")
            elif gpu_use > 80:   alerts.append(f"GPU usage WARNING at {gpu_use:.0f}%")
            if ram > 93:         alerts.append(f"RAM CRITICAL at {ram:.0f}%")
            elif ram > 80:       alerts.append(f"RAM WARNING at {ram:.0f}%")
            if pwr > 12:         alerts.append(f"Power CRITICAL at {pwr:.1f}W")
            elif pwr > 8:        alerts.append(f"Power WARNING at {pwr:.1f}W")
            if volt < volt_fail: alerts.append(f"Voltage CRITICAL at {volt:.2f}V")
            elif volt < volt_warn: alerts.append(f"Voltage WARNING at {volt:.2f}V")

            alert_summary = (
                "⚠️ ACTIVE ALERTS: " + " | ".join(alerts)
                if alerts else "✅ All metrics within normal range"
            )

            # ── Full telemetry context sent to LLM ──
            telemetry_context = f"""
LIVE TELEMETRY — {device_name} ({device_id})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU Temperature  : {cpu_temp:.1f}°C   {'🚨 CRITICAL' if cpu_temp > 82 else '⚠️ WARNING' if cpu_temp > 70 else '✅ OK'}
GPU Temperature  : {gpu_temp:.1f}°C   {'🚨 CRITICAL' if gpu_temp > 84 else '⚠️ WARNING' if gpu_temp > 72 else '✅ OK'}
CPU Usage        : {cpu_use:.1f}%     {'🚨 CRITICAL' if cpu_use > 92 else '⚠️ WARNING' if cpu_use > 75 else '✅ OK'}
GPU Usage        : {gpu_use:.1f}%     {'🚨 CRITICAL' if gpu_use > 95 else '⚠️ WARNING' if gpu_use > 80 else '✅ OK'}
RAM Usage        : {ram:.1f}%         {'🚨 CRITICAL' if ram > 93 else '⚠️ WARNING' if ram > 80 else '✅ OK'}
Power Draw       : {pwr:.1f}W         {'🚨 CRITICAL' if pwr > 12 else '⚠️ WARNING' if pwr > 8 else '✅ OK'}
Voltage          : {volt:.2f}V        {'🚨 CRITICAL' if volt < volt_fail else '⚠️ WARNING' if volt < volt_warn else '✅ OK'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Status   : {status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML HEALTH PREDICTION CARD (shown on dashboard):
  Prediction Label : {ml_pred}
  Confidence Score : {ml_confidence:.1f}%   ← This is what the ML Health Prediction card shows
  Failure Risk     : {ml_failure_risk:.1f}%  ← This is the failure risk bar value
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alert_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Question: {user_message}
"""

            payload = json.dumps({
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": telemetry_context},
                ],
                "max_tokens": 600,
                "temperature": 0.65,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent":    "SENTINEL-AI/1.0",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            reply = result["choices"][0]["message"]["content"]
            print("✅ Groq reply received")
            return jsonify({"reply": reply})

        except Exception as e:
            import traceback
            print(f"🔥 Chat error: {e}")
            traceback.print_exc()
            return jsonify({"reply": _fallback_response(
                data.get("message",  "")    if data else "",
                data.get("telemetry", {})   if data else {},
                data.get("device",  "rpi")  if data else "rpi",
            )}), 200


# ══════════════════════════════════════════════════════
# RULE-BASED FALLBACK
# ══════════════════════════════════════════════════════

def _fallback_response(message, telemetry, device):
    """Smart rule-based fallback — answers ONLY what was asked."""
    q           = message.lower()
    device_name = "Jetson Nano" if device == "jetson" else "Raspberry Pi"
    status      = str(telemetry.get("status", "unknown")).upper()

    # ✅ FIXED: all keys now match app.py /stream-data response
    def tv(key, default=0.0):
        try:    return float(telemetry.get(key, default))
        except: return float(default)

    cpu_temp        = tv("cpu_temp")
    gpu_temp        = tv("gpu_temp")
    cpu_use         = tv("cpu_usage")           # ✅ FIXED: was "cpu_use"
    gpu_use         = tv("gpu_usage")           # ✅ FIXED: was "gpu_use"
    ram             = tv("ram_usage")           # ✅ FIXED: was "ram"
    pwr             = tv("power_consumption")   # ✅ FIXED: was "pwr"
    volt            = tv("voltage")             # ✅ FIXED: was "volt"
    ml_pred         = str(telemetry.get("ml_pred", "unknown")).upper()
    ml_confidence   = tv("ml_confidence",  0)
    ml_failure_risk = tv("ml_failure_risk", 0)

    volt_warn = 7.2 if device == "jetson" else 4.9
    volt_fail = 7.0 if device == "jetson" else 4.7

    def temp_level(v, warn, fail):
        return "🚨 CRITICAL" if v > fail else "⚠️ WARNING" if v > warn else "✅ Normal"

    def use_level(v, warn, fail):
        return "🚨 CRITICAL" if v > fail else "⚠️ WARNING" if v > warn else "✅ Normal"

    # ── GREETINGS ──
    if any(k in q for k in ["hi", "hello", "hey", "hii", "helo", "hiii",
                              "good morning", "good evening", "good afternoon",
                              "howdy", "sup", "what's up", "whats up"]):
        s_emoji = "✅" if status == "OK" else "⚠️" if status == "WARN" else "🚨"
        return (
            f"Hey! 👋 Welcome to <b>SENTINEL AI</b>. "
            f"I'm monitoring your <b>{device_name}</b> — "
            f"currently {s_emoji} <b>{status}</b>.<br><br>"
            f"You can ask me about:<br>"
            f"• 🌡️ <b>Temperature</b> — CPU & GPU thermal status<br>"
            f"• ⚙️ <b>CPU / GPU usage</b> — processing load<br>"
            f"• 🔋 <b>Voltage & Power</b> — supply health<br>"
            f"• 💾 <b>RAM</b> — memory pressure<br>"
            f"• 🧠 <b>ML Prediction</b> — AI health forecast<br>"
            f"What would you like to know? 😊"
        )

    # ── THANKS ──
    if any(k in q for k in ["thank", "thanks", "thx", "ty", "thankyou", "thank you"]):
        return "You're welcome! 😊 I'm always here if you need to check on your device."

    # ── WHO ARE YOU ──
    if any(k in q for k in ["who are you", "what are you", "introduce", "about you"]):
        return (
            f"I'm <b>SENTINEL AI</b> 🤖 — your embedded IoT device health assistant. "
            f"I monitor real-time telemetry from your <b>{device_name}</b> "
            f"and help you detect issues before they become failures. "
            f"Ask me about temperature, CPU/GPU usage, RAM, voltage, ML prediction, or overall health!"
        )

    # ── IRRELEVANT ──
    if any(k in q for k in ["cook", "recipe", "movie", "sport", "weather", "news",
                              "game", "music", "joke", "story", "math", "history",
                              "capital", "country", "president", "cricket", "football",
                              "ipl", "netflix", "amazon", "food", "restaurant", "bitcoin"]):
        return (
            f"Hey! I'm SENTINEL AI — I only know about device health monitoring 😊 "
            f"I can tell you about your <b>{device_name}</b>'s "
            f"temperature, CPU/GPU usage, RAM, voltage, or ML prediction. "
            f"What would you like to check?"
        )

    # ── ML PREDICTION ──
    if any(k in q for k in ["ml", "predict", "prediction", "model",
                              "machine learning", "ai predict", "ai says",
                              "confidence", "confidence score", "failure risk"]):
        emoji = "✅" if ml_pred == "NORMAL" else "⚠️" if ml_pred == "WARNING" else "🚨"
        return (
            f"🧠 <b>ML Health Prediction — {device_name}</b><br><br>"
            f"• <b>Prediction:</b> {emoji} <b>{ml_pred}</b><br>"
            f"• <b>Confidence Score:</b> <b>{ml_confidence:.1f}%</b><br>"
            f"• <b>Failure Risk:</b> <b>{ml_failure_risk:.1f}%</b><br><br>"
            + (
                "✅ The model is confident the device is operating normally."
                if ml_pred == "NORMAL" else
                f"⚠️ The model detects anomaly patterns. Failure risk at <b>{ml_failure_risk:.1f}%</b> — recommend inspection."
                if ml_pred == "WARNING" else
                f"🚨 The model predicts <b>imminent failure</b> with {ml_confidence:.1f}% confidence. Immediate action required!"
            )
        )

    # ── TEMPERATURE ──
    if any(k in q for k in ["temp", "temperature", "temperture", "heat", "hot", "thermal", "overheat"]):
        cpu_s = temp_level(cpu_temp, 70, 82)
        gpu_s = temp_level(gpu_temp, 72, 84)
        resp = (
            f"🌡️ <b>Temperature — {device_name}</b><br><br>"
            f"• <b>CPU Temp:</b> <b>{cpu_temp:.1f}°C</b> — {cpu_s}<br>"
            f"• <b>GPU Temp:</b> <b>{gpu_temp:.1f}°C</b> — {gpu_s}<br><br>"
        )
        if cpu_temp > 82 or gpu_temp > 84:
            resp += "🚨 Temperatures are <b>critically high</b>. Immediate cooling action required!"
        elif cpu_temp > 70 or gpu_temp > 72:
            resp += "⚠️ Temperatures are <b>elevated</b>. Ensure good airflow and check cooling."
        else:
            resp += "✅ Both temperatures are within the <b>safe operating range</b>."
        return resp

    # ── CPU ──
    if any(k in q for k in ["cpu", "processor", "processer", "proccesor", "core usage"]):
        return (
            f"⚙️ <b>CPU — {device_name}</b><br><br>"
            f"• <b>CPU Usage:</b> <b>{cpu_use:.0f}%</b> — {use_level(cpu_use, 75, 92)}<br>"
            f"• <b>CPU Temp:</b> <b>{cpu_temp:.1f}°C</b> — {temp_level(cpu_temp, 70, 82)}<br><br>"
            + (
                "🚨 CPU is under <b>critical load</b>. Reduce running processes immediately."
                if cpu_use > 92 else
                "⚠️ CPU usage is <b>elevated</b>. Consider reducing workload."
                if cpu_use > 75 else
                "✅ CPU is operating <b>normally</b>."
            )
        )

    # ── GPU ──
    if any(k in q for k in ["gpu", "graphics", "cuda", "opencl"]):
        return (
            f"🎮 <b>GPU — {device_name}</b><br><br>"
            f"• <b>GPU Usage:</b> <b>{gpu_use:.0f}%</b> — {use_level(gpu_use, 80, 95)}<br>"
            f"• <b>GPU Temp:</b> <b>{gpu_temp:.1f}°C</b> — {temp_level(gpu_temp, 72, 84)}<br><br>"
            + (
                "🚨 GPU is under <b>critical load</b>. Offload GPU tasks immediately."
                if gpu_use > 95 else
                "⚠️ GPU usage is <b>elevated</b>. Monitor and reduce CUDA workloads."
                if gpu_use > 80 else
                "✅ GPU is operating <b>normally</b>."
            )
        )

    # ── RAM ──
    if any(k in q for k in ["ram", "memory", "memry", "mem", "memmory"]):
        return (
            f"💾 <b>RAM — {device_name}</b><br><br>"
            f"• <b>RAM Usage:</b> <b>{ram:.0f}%</b> — {use_level(ram, 80, 93)}<br>"
            f"• Warning: 80% | Critical: 93%<br><br>"
            + (
                "🚨 Memory is <b>critically full</b>. Kill unused processes immediately."
                if ram > 93 else
                "⚠️ Memory pressure is <b>high</b>. Consider closing background apps."
                if ram > 80 else
                "✅ RAM usage is <b>healthy</b>."
            )
        )

    # ── VOLTAGE / POWER ──
    if any(k in q for k in ["volt", "voltage", "voltge", "power", "watt", "pwr", "supply"]):
        volt_s = (
            "🚨 CRITICAL — dangerously low!" if volt < volt_fail else
            "⚠️ WARNING — below safe threshold" if volt < volt_warn else "✅ Normal"
        )
        pwr_s = "🚨 CRITICAL" if pwr > 12 else "⚠️ WARNING" if pwr > 8 else "✅ Normal"
        return (
            f"🔋 <b>Power — {device_name}</b><br><br>"
            f"• <b>Voltage:</b> <b>{volt:.2f}V</b> — {volt_s}<br>"
            f"• <b>Power Draw:</b> <b>{pwr:.1f}W</b> — {pwr_s}<br><br>"
            + (
                "🚨 Voltage is <b>critically low</b>. Check PSU immediately — hardware damage risk!"
                if volt < volt_fail else
                "⚠️ Voltage is <b>slightly low</b>. Consider a higher-rated PSU."
                if volt < volt_warn else
                "✅ Power supply is <b>stable and healthy</b>."
            )
        )

    # ── OVERALL STATUS ──
    if any(k in q for k in ["status", "health", "overall", "how is", "overview",
                              "summary", "current state", "all metrics",
                              "current condn", "current condition", "condn"]):
        s_emoji  = "✅" if status == "OK" else "⚠️" if status == "WARN" else "🚨"
        ml_emoji = "✅" if ml_pred == "NORMAL" else "⚠️" if ml_pred == "WARNING" else "🚨"
        return (
            f"{s_emoji} <b>{device_name} — {status}</b><br><br>"
            f"• <b>CPU:</b> {cpu_use:.0f}% | {cpu_temp:.1f}°C<br>"
            f"• <b>GPU:</b> {gpu_use:.0f}% | {gpu_temp:.1f}°C<br>"
            f"• <b>RAM:</b> {ram:.0f}%<br>"
            f"• <b>Power:</b> {pwr:.1f}W | <b>Voltage:</b> {volt:.2f}V<br>"
            f"• <b>ML Prediction:</b> {ml_emoji} {ml_pred} ({ml_confidence:.1f}% confidence, {ml_failure_risk:.1f}% failure risk)<br><br>"
            + (
                "✅ All parameters are within safe operating range."
                if status == "OK" else
                "⚠️ One or more parameters need attention. Ask me <b>why</b> or <b>what to do</b>."
                if status == "WARN" else
                "🚨 Critical parameters detected! Immediate action required."
            )
        )

    # ── WHY / CAUSES ──
    if any(k in q for k in ["why", "reason", "cause", "issue", "problem",
                              "what happened", "whats wrong", "fail", "warning", "alert", "anomaly"]):
        issues = []
        if cpu_temp > 82:      issues.append(f"🚨 CPU Temp <b>{cpu_temp:.1f}°C</b> — critical (limit: 82°C)")
        elif cpu_temp > 70:    issues.append(f"⚠️ CPU Temp <b>{cpu_temp:.1f}°C</b> — warning (limit: 70°C)")
        if gpu_temp > 84:      issues.append(f"🚨 GPU Temp <b>{gpu_temp:.1f}°C</b> — critical (limit: 84°C)")
        elif gpu_temp > 72:    issues.append(f"⚠️ GPU Temp <b>{gpu_temp:.1f}°C</b> — warning (limit: 72°C)")
        if cpu_use > 92:       issues.append(f"🚨 CPU Usage <b>{cpu_use:.0f}%</b> — critical (limit: 92%)")
        elif cpu_use > 75:     issues.append(f"⚠️ CPU Usage <b>{cpu_use:.0f}%</b> — warning (limit: 75%)")
        if gpu_use > 95:       issues.append(f"🚨 GPU Usage <b>{gpu_use:.0f}%</b> — critical (limit: 95%)")
        elif gpu_use > 80:     issues.append(f"⚠️ GPU Usage <b>{gpu_use:.0f}%</b> — warning (limit: 80%)")
        if ram > 93:           issues.append(f"🚨 RAM <b>{ram:.0f}%</b> — critical (limit: 93%)")
        elif ram > 80:         issues.append(f"⚠️ RAM <b>{ram:.0f}%</b> — warning (limit: 80%)")
        if pwr > 12:           issues.append(f"🚨 Power <b>{pwr:.1f}W</b> — critical (limit: 12W)")
        elif pwr > 8:          issues.append(f"⚠️ Power <b>{pwr:.1f}W</b> — warning (limit: 8W)")
        if volt < volt_fail:   issues.append(f"🚨 Voltage <b>{volt:.2f}V</b> — critical (min: {volt_fail}V)")
        elif volt < volt_warn: issues.append(f"⚠️ Voltage <b>{volt:.2f}V</b> — warning (min: {volt_warn}V)")

        if not issues:
            return (f"✅ <b>{device_name}</b> looks perfectly healthy! "
                    f"No threshold breaches detected. ML also predicts: <b>{ml_pred}</b>.")

        ml_note = ""
        if ml_pred in ("WARNING", "FAILURE"):
            ml_note = f"<br><br>🧠 ML model also predicts <b>{ml_pred}</b> with <b>{ml_confidence:.1f}%</b> confidence ({ml_failure_risk:.1f}% failure risk)."

        return (
            f"🔍 <b>Issues on {device_name}:</b><br><br>"
            + "<br>".join(f"  {i}" for i in issues)
            + ml_note
        )

    # ── FIX / WHAT TO DO ──
    if any(k in q for k in ["fix", "recommend", "solution", "precaution",
                              "what to do", "what should", "help", "suggestion",
                              "action", "repair", "resolve", "how to fix"]):
        if status == "OK":
            return (
                "✅ <b>All systems nominal!</b> No action needed right now.<br><br>"
                "Best practices:<br>"
                "① Ensure good airflow around the device<br>"
                "② Keep heatsink clean and dust-free<br>"
                "③ Avoid running heavy tasks 24/7<br>"
                "④ Monitor voltage if using a cheap PSU"
            )
        recs = []
        if cpu_temp > 70 or gpu_temp > 72:
            recs += ["🌡️ <b>Thermal issues:</b>",
                     "① Clean dust from heatsink and fan immediately",
                     "② Reapply thermal paste if temps stay high",
                     "③ Ensure 10cm clearance around the device",
                     "④ Add active cooling fan if not present"]
        if cpu_use > 75 or gpu_use > 80:
            recs += ["⚙️ <b>High load:</b>",
                     "① Kill non-essential processes (use <code>htop</code>)",
                     "② Reduce inference batch size for ML workloads",
                     "③ Schedule heavy tasks during off-peak hours"]
        if ram > 80:
            recs += ["💾 <b>Memory pressure:</b>",
                     "① Restart services with high memory usage",
                     "② Add swap: <code>sudo dphys-swapfile swapoff && sudo nano /etc/dphys-swapfile</code>",
                     "③ Profile your app for memory leaks"]
        if volt < volt_warn or pwr > 8:
            recs += ["🔋 <b>Power issues:</b>",
                     "① Replace PSU with official recommended supply",
                     "② Use short thick USB cables",
                     "③ Disconnect non-essential USB peripherals"]
        if not recs:
            recs = ["① Reduce workload by 20–30%",
                    "② Check cooling and airflow",
                    "③ Monitor all metrics for the next 30 minutes"]

        ml_note = ""
        if ml_pred in ("WARNING", "FAILURE"):
            ml_note = f"<br><br>🧠 ML model is predicting <b>{ml_pred}</b> ({ml_confidence:.1f}% confidence) — act promptly."

        return (
            f"🛠️ <b>Recommendations for {device_name}:</b><br><br>"
            + "<br>".join(recs) + ml_note
        )

    # ── TIME TO FAILURE ──
    if any(k in q for k in ["time", "how long", "until", "ttf", "when will",
                              "failure time", "estimate", "how much time"]):
        if status == "OK":
            return (f"✅ <b>{device_name}</b> is healthy — no failure predicted. "
                    f"ML confirms: <b>{ml_pred}</b> ({ml_confidence:.1f}% confidence).")
        if status == "WARN":
            return (f"⚠️ Estimated time to failure: <b>4–12 hours</b> at current load.<br>"
                    f"ML failure risk: <b>{ml_failure_risk:.1f}%</b>. Recommend reducing workload now.")
        return (f"🚨 <b>{device_name}</b> is critical. Failure within <b>30 min – 2 hours</b>.<br>"
                f"ML predicts <b>{ml_pred}</b> with <b>{ml_confidence:.1f}%</b> confidence. <b>Act immediately!</b>")

    # ── DEFAULT ──
    return (
        f"I'm monitoring <b>{device_name}</b> — currently <b>{status}</b>.<br><br>"
        f"Ask me about:<br>"
        f"• 🌡️ Temperature | ⚙️ CPU/GPU | 💾 RAM | 🔋 Voltage<br>"
        f"• 🧠 ML Prediction | 🔍 Why issues? | 🛠️ What to do?<br><br>"
        f"How can I help? 😊"
    )