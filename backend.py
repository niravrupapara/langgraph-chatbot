from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage ,SystemMessage, HumanMessage , AIMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from langgraph.prebuilt import ToolNode , tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun

import requests


import sqlite3

from langgraph.graph.message import add_messages
from mistralai import Mistral
from dotenv import load_dotenv

import os


load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = os.getenv("MISTRAL_MODEL")





# -------------------
# 2. Tools
# -------------------
# Tools



search_tool = DuckDuckGoSearchRun(region='en-us')


@tool
def get_stock_price(symbol : str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    return data



@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    




tools = [search_tool, get_stock_price, calculator]



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
    messages = state["messages"]

    llm = ChatMistralAI(
        model=MODEL,
        api_key=API_KEY,
    )

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}



connection = sqlite3.connect("chatbot_state.db" , check_same_thread=False)

checkpointer = SqliteSaver(conn=connection)

def retrive_all_states():
    all_states = set()

    for checkpoint in checkpointer.list(None):
        all_states.add(checkpoint.config['configurable']['thread_id'])

    return list(all_states)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_with_mistral)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")  

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

graph.add_edge("chat_node", END)


chatbot = graph.compile(checkpointer=checkpointer)


