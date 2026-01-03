from dataclasses import dataclass
from typing import Any, Callable
import time

@dataclass
class ToolResult:
    """Structured result from tool execution"""
    success: bool
    data: Any
    error: str | None = None
    execution_time: float | None = None
    
    def __str__(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        return str(self.data)


def execute_with_metrics(
    tool_name: str,
    func: Callable[..., ToolResult],
    **kwargs
) -> str:
    """Execute a tool function with timing and error handling"""
    start = time.time()
    
    try:
        result = func(**kwargs)
    except Exception as e:
        result = ToolResult(success=False, data=None, error=str(e))
    
    result.execution_time = time.time() - start
    
    # TODO: Add logging/metrics here
    # logger.info(f"{tool_name}: success={result.success}, time={result.execution_time:.2f}s")
    
    return str(result)