from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr
import os
import time
from openai import InternalServerError, RateLimitError, APIConnectionError

load_dotenv(override=True)

MODEL_NAME =  "gemini-3.1-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)



system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]


def chat(message, history):
    messages = system + history + [
        {"role": "user", "content": message}
    ]

    for attempt in range(3):
        try:
            response = gemini.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                timeout=60
            )

            while response.choices[0].finish_reason == "tool_calls":
                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls

                results = handle_tool_calls(tool_calls)

                messages.append(assistant_message)
                messages.extend(results)

                response = gemini.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    timeout=60
                )

            content = response.choices[0].message.content

            return content or "I couldn't generate a response. Please try again."

        except (InternalServerError, RateLimitError, APIConnectionError) as e:
            print(f"Gemini API error (attempt {attempt + 1}): {e}")

            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return (
                    "The AI service is temporarily busy. "
                    "Please try again in a few moments."
                )

        except Exception as e:
            print(f"Unexpected error: {e}")
            return "Something went wrong. Please try again."


if __name__ == "__main__":
    demo = gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    )

    demo.launch(
        css=CSS,
        js=JS,
        theme=gr.themes.Base(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
    )
