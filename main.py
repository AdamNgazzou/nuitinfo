import os
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

# Interactive chat loop: user types a prompt and the model responds. Type 'exit' or 'quit' to stop.
try:
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
            )
        except ClientError as e:
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            if status == 429 or (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429):
                print("Quota exceeded: check billing/quotas in Google Cloud Console")
                break
            print(f"API error: {e}")
            continue
        except KeyboardInterrupt:
            print("\nInterrupted, exiting.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

        # Print response text when available.
        assistant_text = getattr(response, "text", None) or str(response)
        print("AI:", assistant_text)
except KeyboardInterrupt:
    print("\nExiting.")