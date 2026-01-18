from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage ,SystemMessage, HumanMessage , AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.graph.message import add_messages
from mistralai import Mistral
from dotenv import load_dotenv

import os


load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = os.getenv("MISTRAL_MODEL")

client = Mistral(api_key=API_KEY)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




def convert_messages(messages):
    converted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            continue

        converted.append({"role": role, "content": msg.content})
    return converted

# ---- Node ----
def chat_with_mistral(state: ChatState):
    messages = convert_messages(state["messages"])

    response = client.chat.complete(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=512
    )

    answer = response.choices[0].message.content

    return {
        "messages": [AIMessage(content=answer)]
    }



connection = sqlite3.connect("chatbot_state.db" , check_same_thread=False)

checkpointer = SqliteSaver(conn=connection)

def retrive_all_states():
    all_states = set()

    for checkpoint in checkpointer.list(None):
        all_states.add(checkpoint.config['configurable']['thread_id'])

    return list(all_states)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_with_mistral)
graph.add_edge(START, "chat_node")  
graph.add_edge("chat_node", END)


chatbot = graph.compile(checkpointer=checkpointer)


