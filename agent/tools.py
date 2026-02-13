from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from typing import Callable

from .memory import MemoryStore
from .types import ToolResult


class SafeEvaluator(ast.NodeVisitor):
    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        op = self.OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return float(op(left, right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        op = self.OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return float(op(self.visit(node.operand)))

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise ValueError("constants must be numbers")
        return float(node.value)

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported syntax: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return SafeEvaluator().visit(tree)


class ToolRegistry:
    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory
        self._tools: dict[str, Callable[[dict], ToolResult]] = {
            "now_time": self._now_time,
            "calculator": self._calculator,
            "recall_user_profile": self._recall_user_profile,
        }

    @property
    def allowed_tool_names(self) -> set[str]:
        return set(self._tools.keys())

    async def execute(self, name: str, args: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, output="unknown tool")
        try:
            result = tool(args)
            if result.name == "recall_user_profile":
                user_id = int(args.get("user_id", 0))
                facts = await self._memory.get_profile_facts(user_id=user_id, limit=8)
                return ToolResult(name=name, ok=True, output="\n".join(facts) if facts else "no facts")
            return result
        except Exception as exc:
            return ToolResult(name=name, ok=False, output=f"tool error: {exc}")

    def _now_time(self, _: dict) -> ToolResult:
        now = datetime.now(timezone.utc).isoformat()
        return ToolResult(name="now_time", ok=True, output=f"UTC time: {now}")

    def _calculator(self, args: dict) -> ToolResult:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return ToolResult(name="calculator", ok=False, output="missing expression")
        value = safe_calculate(expression)
        return ToolResult(name="calculator", ok=True, output=f"{expression} = {value}")

    def _recall_user_profile(self, args: dict) -> ToolResult:
        user_id = int(args.get("user_id", 0))
        if user_id <= 0:
            return ToolResult(name="recall_user_profile", ok=False, output="invalid user_id")
        return ToolResult(name="recall_user_profile", ok=True, output="fetching profile facts")
