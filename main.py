import os
import time
from collections import deque
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# Load .env and read API key. Accept both GEMINI_API_KEY and GOOGLE_API_KEY as a fallback.
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not set in .env or environment")

# Construct the client. Prefer passing the key explicitly; if the client API does not accept
# an api_key parameter, set the env var and construct the client normally.
try:
    client = genai.Client(api_key=API_KEY)
except TypeError:
    os.environ["GEMINI_API_KEY"] = API_KEY
    client = genai.Client()

# Simple in-memory rate limiter using a sliding window of timestamps.
class InMemoryLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = int(max_calls)
        self.period = float(period)
        self.calls = deque()

    def _prune(self):
        cutoff = time.time() - self.period
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

    def allow(self) -> tuple[bool, float]:
        """Return (allowed, wait_seconds)."""
        self._prune()
        if len(self.calls) < self.max_calls:
            return True, 0.0
        wait = self.calls[0] + self.period - time.time()
        return False, max(0.0, wait)

    def record(self) -> None:
        """Record a request timestamp (should be called when actually sending a request)."""
        self._prune()
        self.calls.append(time.time())

# Configure limiter from environment (defaults: 20 calls per 60 seconds)
RATE_MAX = int(os.getenv("RATE_LIMIT_MAX", 20))
RATE_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
limiter = InMemoryLimiter(RATE_MAX, RATE_WINDOW)

# Interactive chat loop: user types a prompt and the model responds. Type 'exit' or 'quit' to stop.
try:
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        allowed, wait = limiter.allow()
        if not allowed:
            print(f"Rate limit reached: waiting {wait:.1f}s (max {RATE_MAX} per {RATE_WINDOW}s)")
            time.sleep(wait)

        # Attempt request with simple exponential backoff on 429.
        attempt = 0
        max_attempts = 5
        while True:
            attempt += 1
            try:
                limiter.record()
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_input,
                )
                break
            except ClientError as e:
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                if status == 429 or (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429):
                    # On quota errors, backoff exponentially and retry up to max_attempts.
                    if attempt >= max_attempts:
                        print("Quota exceeded after retries: check billing/quotas in Google Cloud Console")
                        response = None
                        break
                    sleep_for = min(60, (2 ** attempt))
                    print(f"Quota/429 received, backing off {sleep_for}s (attempt {attempt}/{max_attempts})")
                    time.sleep(sleep_for)
                    continue
                else:
                    print(f"API error: {e}")
                    response = None
                    break
            except KeyboardInterrupt:
                print("\nInterrupted, exiting.")
                raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                response = None
                break

        if response is None:
            # skip printing if we failed
            continue

        # Print response text when available.
        assistant_text = getattr(response, "text", None) or str(response)
        print("AI:", assistant_text)
except KeyboardInterrupt:
    print("\nExiting.")