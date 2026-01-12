from pathlib import Path
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, SystemMessage

from codecompass.tools import create_tools
from codecompass.config import settings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig 

import json
    
# Debugging
from pydantic import PrivateAttr
from rich.console import Console


SYSTEM_PROMPT = """You are CodeCompass, an AI assistant that helps developers understand and navigate codebases accurately.

## Tool Selection Guidelines

Choose the most appropriate tool:

- **search_code**: Conceptual queries when no exact symbol or file is specified. Keep queries SHORT (2-4 keywords). Remove filler words. 
- **find_references**: When you KNOW the exact symbol name. Extract just the symbol. Use **find_references** whenever the user asks what would break, what is affected, or where a symbol is used, even if they phrase it as an impact or dependency question.
- **get_git_history**: Questions about changes, history, who modified, when changed.
- **read_file**: When you know the exact file path.
- **get_file_structure**: Project overview or finding file locations.
- **get_dependencies**: For understanding what a file imports.

Examples:
- "How is authentication implemented?"
  → search_code("authentication")
  → read_file("src/auth/AuthService.py")
  → explain implementation

- "What would break if I change UserService?"
  → find_references("UserService")
  → read_file("src/services/UserService.py")
  → summarize impact

- "What changed in src/auth recently?"
  → get_git_history("src/auth")
  → summarize recent changes

## Workflow Rules
1. For "how is X used" questions: find_references → read_file (1-3 files) → explain. 
2. Many find_references results (5+): summarize patterns, read_file only representative examples.
3. search_code isn't exhaustive. Follow up with read_file or find_references for "how/why" questions.
4. find_references is for locating call sites only. Do not explain behavior or include code blocks based solely on find_references; you must read_file the defining file and/or the callsite file first.

## Core Rules (CRITICAL)

### 1. Read Before Explaining
Never explain what code does based on snippets, call sites, or imports alone. A symbol is "known" only after read_file returns its full definition body. If you lack the definition, call read_file immediately. Do not ask permission or say "Would you like me to...?"

### 2. Verbatim Code Only
Every code block in your response must appear exactly in tool output. Do not reconstruct or infer code.

### 3. Answer the Question Asked
Always respond in direct accordance with the user's question. Avoid tangents or unrequested explanations. Only provide details relevant to the user's specific query.

## Response Guidelines
- Be concise but thorough: explain each function in your own words (purpose → inputs/deps → key behavior → outputs/errors). Do not simply paste or restate code blocks without justification.
- Reference specific files and line numbers.
- If initial tool results are insufficient, try a different tool or search query. Combine multiple tools (e.g., `read_file` + `find_references`) if needed to fully answer the question.
- After using tools as needed to answer the question, synthesize the information into a clear answer.
"""

# For debugging purposes (specifically printing langgraph generated tool descriptions)
class DebugChatOllama(ChatOllama):
    """ChatOllama wrapper that logs first request and tracks token usage"""

    _printed_once: bool = PrivateAttr(default=False)
    _prev_context_size: int = PrivateAttr(default=0)
    _cumulative_tokens: int = PrivateAttr(default=0)  # actual tokens spent across all calls

    def generate(self, messages, **kwargs):
        kwargs.pop("run_manager", None)
        console = Console(markup=False)

        if not self._printed_once:
            self._printed_once = True
            flat_messages = self._flatten_messages(messages)
            console.print("=" * 60, style="bold yellow")
            console.print("FULL REQUEST TO OLLAMA (FIRST CALL ONLY)", style="bold yellow")
            console.print("=" * 60, style="bold yellow")
            for msg in flat_messages:
                role = self._get_role(msg)
                content = getattr(msg, "content", str(msg))
                console.print(f"[{role.upper()}]: {content}", style="yellow", highlight=False)
            if "tools" in kwargs:
                console.print("\n--- TOOLS ---", style="bold yellow")
                console.print(json.dumps(kwargs["tools"], indent=2), style="yellow", highlight=False)
            console.print("=" * 60 + "\n", style="bold yellow")

        result = super().generate(messages, **kwargs)

        ai_msg = result.generations[0][0].message
        usage = getattr(ai_msg, "usage_metadata", None)

        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

            # Calculate new tokens added this turn
            if input_tokens < self._prev_context_size:
                # Context was truncated
                new_input = input_tokens
                truncated = True
            else:
                new_input = input_tokens - self._prev_context_size
                truncated = False

            self._prev_context_size = input_tokens + output_tokens
            self._cumulative_tokens += new_input + output_tokens

            truncate_flag = " (truncated!)" if truncated else ""
            console.print(
                f"\n\n[TOKENS] new={new_input}{truncate_flag} | out={output_tokens} | "
                f"total={total_tokens} | spent={self._cumulative_tokens}",
                style="yellow"
            )
        else:
            console.print("[TOKENS] usage_metadata missing", style="yellow")

        return result

    def _flatten_messages(self, messages):
        flat = []
        for m in messages:
            if isinstance(m, list):
                flat.extend(self._flatten_messages(m))
            else:
                flat.append(m)
        return flat

    def _get_role(self, msg: BaseMessage) -> str:
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )

        if isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        elif isinstance(msg, SystemMessage):
            return "system"
        elif isinstance(msg, ToolMessage):
            return "tool"
        return "unknown"

def create_agent(repo_path: Path, debug: bool = False):
    """Create the CodeCompass agent"""
    
    tools = create_tools(repo_path)
    # llm = ChatOllama(model=settings.chat_model).bind_tools(tools)
    if debug:
        llm = DebugChatOllama(model=settings.chat_model, num_ctx=settings.ollama_ctx_window).bind_tools(tools)
    else:
        llm = ChatOllama(model=settings.chat_model, num_ctx=settings.ollama_ctx_window).bind_tools(tools)


    def reason(state: MessagesState, config: RunnableConfig):  
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        response = llm.invoke(messages, config=config)
        return {"messages": [response]}
    
    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    graph = StateGraph(MessagesState)
    graph.add_node("reason", reason)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "reason")
    
    return graph.compile()


class CodeCompassAgent:
    """Wrapper for conversation management"""
    
    def __init__(self, repo_path: Path, use_memory: bool = True, debug: bool = False):
        self.repo_path = Path(repo_path)
        self.graph = create_agent(self.repo_path, debug=debug)
        self.use_memory = use_memory
        self.messages: list = []
        self.last_state: MessagesState | None = None
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a response"""
        if self.use_memory:
            self.messages.append(HumanMessage(content=user_message))
            input_messages = self.messages
        else:
            input_messages = [HumanMessage(content=user_message)]
        
        result = self.graph.invoke({"messages": input_messages})
        
        if self.use_memory:
            self.messages = result["messages"]
        
        return result["messages"][-1].content
    
    async def chat_stream_async(self, user_message: str):
        """True streaming with token-level events"""

        if self.use_memory:
            self.messages.append(HumanMessage(content=user_message))
            input_messages = self.messages
        else:
            input_messages = [HumanMessage(content=user_message)]
        
        final_messages = input_messages.copy()
        
        async for event in self.graph.astream_events(
            {"messages": input_messages},
            version="v2"
        ):
            kind = event["event"]
            
            if kind == "on_chat_model_start":
                yield {"type": "thinking", "status": "started"}
            elif kind == "on_chat_model_stream":
                # Token-by-token streaming
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content}
            elif kind == "on_tool_start":
                yield {"type": "tool_call", "tool": event["name"]}
            elif kind == "on_tool_end":
                yield {"type": "tool_result", "content": str(event["data"])[:200]}
            elif kind == "on_chain_end":
                if "messages" in event["data"].get("output", {}):
                    final_messages = event["data"]["output"]["messages"]
                    self.last_state = event["data"]["output"]

        if self.use_memory:
            self.messages = final_messages
    
    def reset(self):
        """Clear conversation history"""
        self.messages = []

    def save_state(self, path: Path):
        from datetime import datetime

        if self.last_state is None:
            return  # nothing to save

        def message_to_dict(msg: BaseMessage) -> dict:
            if hasattr(msg, "model_dump"):
                return msg.model_dump()
            return msg.dict()

        state = {
            "repo_path": str(self.repo_path),
            "use_memory": self.use_memory,
            "saved_at": datetime.utcnow().isoformat(),
            "messages": [
                message_to_dict(m)
                for m in self.last_state["messages"]
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
