# test_groq.py  — run this directly: python test_groq.py
import os, json, urllib.request, urllib.error
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

api_key = os.environ.get("GROQ_API_KEY", "").strip()

print("=" * 50)
print("KEY PREFIX  :", api_key[:8]  if api_key else "MISSING")
print("KEY LENGTH  :", len(api_key))
print("STARTS gsk_ :", api_key.startswith("gsk_"))
print("=" * 50)

if not api_key:
    print("❌ No API key found. Check your .env file.")
    exit(1)

payload = json.dumps({
     "model": "llama-3.1-8b-instant", 
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
}).encode("utf-8")

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "SENTINEL-AI/1.0",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    print("✅ SUCCESS! Reply:", result["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8")
    print(f"❌ HTTP {e.code}: {e.reason}")
    print("Response body:", body)
except Exception as e:
    print(f"❌ Error: {e}")