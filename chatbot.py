# chatbot_core.py

import os
from dotenv import load_dotenv

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage

from typing_extensions import TypedDict
from typing import Annotated

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---------- TOOLS ----------

# Arxiv Tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Search academic research papers from Arxiv")

# Wikipedia Tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Search articles from Wikipedia")

# Tavily Tool
tavily_tool = TavilySearchResults()

tools = [arxiv_tool, wikipedia_tool, tavily_tool]

# ---------- LLM ----------

llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
llm_with_tools = llm.bind_tools(tools=tools)
 

# ---------- LangGraph Nodes ----------

def tool_calling_llm(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ---------- LangGraph Construction ----------

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")  # loop back if tools used

graph = builder.compile()

# ---------- Main Function ----------

def generate_response(user_query: str):
    """
    Call this function in Streamlit to get the chatbot's response using LangGraph.
    """
    from langchain_core.messages import HumanMessage

    result = graph.invoke({"messages": [HumanMessage(content=user_query)]})
    
    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai":
            return msg.content
    return "Sorry, I couldn't understand that."