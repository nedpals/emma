import ast
import operator

from tools import Tool, ToolResult

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculateTool(Tool):
    name = "calculate"
    description = "Safely evaluate a mathematical expression and return the numeric result. Supports +, -, *, /, //, %, **."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate, e.g. '3 * 8 + 2'",
            },
        },
        "required": ["expression"],
    }

    def execute(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return ToolResult(content=str(result), success=True)
        except Exception as e:
            return ToolResult(content=f"Error: {e}", success=False)
