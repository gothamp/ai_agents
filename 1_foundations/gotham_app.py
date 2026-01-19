# 
# Tool use with AgenticAI.
# Using Pushover for push notifications

# 
# imports
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from PyPDF2 import PdfReader
import gradio as gr

# 
# setup llm 
load_dotenv(override=True)


# 
# setup pushover
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


#
def push(message):
    """Send a push notification via Pushover API."""
    print(f"Push : {message}")

    payload = {
        "token": pushover_token,
        "user": pushover_user,
        "message": message,
        "title": "Gotham Lab 4"
    }
    requests.post(
        pushover_url,
        data=payload,
        timeout=10
    )


def record_user_details(email, name="Name not provided", notes="Notes not provided"):
    """
    Record details of an interested user and send a push notification.

    Args:
        email (str): The user's email address.
        name (str, optional): The user's name. Defaults to "Name not provided".
        notes (str, optional): Additional notes. Defaults to "Notes not provided".

    Returns:
        dict: Confirmation that the details were recorded.
    """
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    """Record a question that couldn't be answered and send a push notification."""
    push(f"Recording {question} asked that I don't know the answer to")
    return "Unknown Question recorded successfully"


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that an user is interested in getting touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name , if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional info about the conversation that the user provided"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}


record_unknown_question_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that an user is interested in getting touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that could not be answered"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}


tools = [{"type": "function", "function": record_user_details_json}, {"type": "function", "function": record_unknown_question_json}]


class Me:
    """Represents the chatbot persona with profile data and LLM interactions."""

    def __init__(self) -> None:
        self.openai = OpenAI()
        self.name = "Gotham"
        reader = PdfReader("gotham/linkedin_profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("gotham/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
# 

    def handle_tool_calls(self, tool_calls):
        """Process tool calls from the LLM and return results."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool call: {tool_name} with arguments: {arguments}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else None
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def get_system_prompt(self):
        """Build and return the system prompt for the chatbot."""
        prompt = f"""You are acting as {self.name}. You are answering questions on {self.name}'s website,
particularly questions related to {self.name}'s career, background, skills and experience.
Please greet the user as soon as you start. Address yourself as Gotham when speaking to anyone.
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible.
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career.
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool.
In general try to get the user to contact me on email or provide his email address so that I can contact him"""
        prompt += f"Here is the user's LinkedIn profile: {self.linkedin}"
        prompt += f"Here is the user's LinkedIn summary: {self.summary}"
        prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return prompt

    def chat(self, message, history):
        """Handle chat interactions with the LLM, including tool calls."""
        messages = [{"role": "system", "content": self.get_system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(
        me.chat, 
        type="messages",
        chatbot=gr.Chatbot(
            type="messages",
            value=[{
                "role": "assistant",
                "content": (
                    "Hi! I'm Gotham. Feel free to ask me about my background, "
                    "skills, or experience!"
                )
            }]
        )
    ).launch(share=True)

