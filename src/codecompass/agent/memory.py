"""
Simple conversation memory management that performs:
- Tool result compression
- Sliding window if context window filled
"""

import re
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage


def compress_tool_result(tool_msg: ToolMessage) -> ToolMessage:
    """Compress a tool result based on tool type"""
    content = tool_msg.content
    name = tool_msg.name
    
    if name == "read_file":
        lines = content.split('\n')
        filename_match = re.match(r'\*\*(.+?)\*\*', lines[0]) if lines else None
        filename = filename_match.group(1) if filename_match else "file"
        functions = re.findall(r'def (\w+)\(', content)
        classes = re.findall(r'class (\w+)', content)
        
        parts = [f"[FILE: {filename}, {len(lines)} lines]"]
        if classes:
            parts.append(f"Classes: {', '.join(classes[:5])}")
        if functions:
            parts.append(f"Functions: {', '.join(functions[:8])}")
        compressed = " | ".join(parts)
    
    elif name == "search_code":
        matches = re.findall(r'\*\*(.+?)\*\*', content)
        compressed = f"[SEARCH: {len(matches)} results - {', '.join(matches[:5])}]"
    
    elif name == "get_file_structure":
        lines = content.split('\n')
        compressed = f"[FILE STRUCTURE: {len(lines)} entries]"
    
    elif name == "find_references":
        ref_count = content.count("Line ")
        compressed = f"[REFERENCES: {ref_count} usages found]"
    
    else:
        compressed = f"[{name}: {len(content)} chars]"
    
    return ToolMessage(
        content=compressed,
        tool_call_id=tool_msg.tool_call_id,
        name=tool_msg.name,
    )


def prepare_messages(
    messages: list[BaseMessage],
    current_tokens: int,
    max_tokens: int = 4096,
    keep_recent_tools: int = 2,
    max_tool_result_tokens: int = 1000
) -> list[BaseMessage]:
    """
    Prepare messages: compress old and long tools, apply sliding window if needed.
    
    Args:
        messages: Full conversation history
        current_tokens: Actual token count from last usage_metadata
        max_tokens: Max tokens before triggering compression
        keep_recent_tools: Keep N most recent tool results uncompressed
    
    Returns:
        Processed messages
    """
    # Separate system messages
    system = [m for m in messages if isinstance(m, SystemMessage)]
    conversation = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Step 1: Compress old + large tool results 
    tool_indices = [i for i, m in enumerate(conversation) if isinstance(m, ToolMessage)]
    recent_tools = set(tool_indices[-keep_recent_tools:]) if tool_indices else set()
    
    processed = []
    for i, msg in enumerate(conversation):
        if isinstance(msg, ToolMessage):
            is_old = i not in recent_tools
            is_too_large = len(msg.content) > max_tool_result_tokens * 4  # Rough char estimate
            
            if is_old or is_too_large:
                processed.append(compress_tool_result(msg))
            else:
                processed.append(msg)
        else:
            processed.append(msg)
    
    # Step 2: If over limit, drop oldest turns
    if current_tokens > max_tokens:
        while len(processed) > 2:
            # Drop oldest turn (find first HumanMessage and everything until next HumanMessage)
            while processed and not isinstance(processed[0], HumanMessage):
                processed.pop(0)
            if processed:
                processed.pop(0)  # Drop the HumanMessage
            while processed and not isinstance(processed[0], HumanMessage):
                processed.pop(0)
            
            # After dropping one turn, check if we've dropped enough (rough 30%)
            if len(processed) < len(conversation) * 0.7:
                break
    
    return system + processed