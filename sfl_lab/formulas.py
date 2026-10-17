"""A bounded numerical expression language for explicitly typed SFL hook points.

Expressions are parsed into Python's AST, but are never evaluated as Python code.
Only the nodes and numerical operations enumerated here can be interpreted.
Validation uses representative tensors; the same checks also run on every real
hook result because a finite collection of probes is not a mathematical proof.
"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from functools import lru_cache
from typing import Any

import torch

from .config import CandidateConfig


class FormulaError(ValueError):
    """An expression cannot safely execute under the selected hook contract."""


LIMITS = {"expression_characters": 2000, "ast_nodes": 256, "depth": 24, "operations": 80}
_HOOK_DATA = [
    ("H1", "Client forward processing", "A", "A", ["A"], "Transform the client-front output before transmission."),
    ("H2", "Activation processing / protection", "A", "A", ["A"], "Protect the client activation at its transmission boundary."),
    ("H3", "Server forward processing", "Z", "Z", ["Z"], "Transform the server output before the client-back computation."),
    ("H4", "Loss function", "loss", "loss", ["loss"], "Transform a client-local scalar loss, preserving its gradient."),
    ("H5", "Gradient calculation", "G", "G", ["G"], "Transform the computed client-back gradient."),
    ("H6", "Gradient processing / protection", "G", "G", ["G"], "Protect the gradient before it leaves the client back."),
    ("H7", "Local optimization", "eta", "eta", ["eta"], "Set a positive scalar learning rate for the configured optimizer."),
    ("H8", "Federated aggregation", "n / N", "weights", ["weights", "n", "N"], "Return a probability vector of client aggregation weights."),
    ("H9", "Privacy accounting diagnostic", "rho", "rho", ["rho", "alpha", "delta"], "Explore an accounting expression; the certified accountant remains independent."),
    ("H10", "Split selection", "score", "score", ["score", "S", "C", "latency", "energy"], "Score feasible split profiles; the engine selects among measured candidates."),
    ("H11", "Communication model", "S, R, p, d", "seconds", ["S", "R", "p", "d"], "Estimate transfer time from bits, bits/second, loss probability and delay seconds."),
    ("H12", "Energy model", "energy", "joules", ["energy", "E", "E_tx", "E_rx", "E_compute"], "Transform the estimated energy in joules."),
    ("H13", "Client selection", "scores", "scores", ["scores", "n"], "Score clients; the engine applies the configured participation count."),
    ("H14", "Non-IID correction", "w, w_global", "regularizer", ["regularizer", "w", "w_global"], "Return a differentiable nonnegative local regularization penalty."),
]
HOOKS = {row[0]: dict(zip(["id", "name", "input", "output", "variables", "description"], row)) for row in _HOOK_DATA}
_CONTEXT = {key: set(item["variables"]) for key, item in HOOKS.items()}
_RESERVED = set().union(*_CONTEXT.values()) | {"generator", "pi", "e"}
_FUNCTION_ARITY = {
    "clip": (2, 2), "gaussian": (3, 3), "noise": (2, 2), "laplace": (2, 2),
    "abs": (1, 1), "norm": (1, 1), "sqrt": (1, 1), "log": (1, 1), "exp": (1, 1),
    "sum": (1, 1), "mean": (1, 1), "var": (1, 1), "min": (2, 2), "max": (2, 2),
    "relu": (1, 1), "sigmoid": (1, 1), "tanh": (1, 1), "softmax": (1, 1),
    "normalize": (1, 1), "clamp": (3, 3), "dot": (2, 2),
}
_RESERVED |= set(_FUNCTION_ARITY)
_OP_DATA = [
    ("CLIENT_FORWARD", 10, True, {}, "Client front: raw features → activation."),
    ("ACTIVATION_CLIP", 11, False, {"C_A": 1.0}, "Clip activation L2 norm to C_A > 0."),
    ("ACTIVATION_GAUSSIAN", 12, False, {"sigma_A": 1.0}, "Add Gaussian noise with standard deviation sigma_A × C_A; requires activation clipping."),
    ("SEND_TO_SERVER", 20, True, {}, "Detach and send the activation boundary tensor."),
    ("SERVER_FORWARD", 30, True, {}, "Run the server middle layers."),
    ("SEND_TO_CLIENT_BACK", 40, True, {}, "Detach and return the server output to the client."),
    ("CLIENT_BACK_FORWARD", 50, True, {}, "Run client-back layers on the returned tensor."),
    ("CALCULATE_LOSS", 60, True, {}, "Calculate the loss using labels retained at the client."),
    ("BACKWARD", 70, True, {}, "Differentiate the client-back loss."),
    ("GRADIENT_CLIP", 71, False, {"C_G": 1.0}, "Clip the outgoing client-back gradient L2 norm to C_G > 0."),
    ("GRADIENT_GAUSSIAN", 72, False, {"sigma_G": 1.0}, "Add Gaussian noise with standard deviation sigma_G × C_G; requires gradient clipping."),
    ("SERVER_BACKWARD", 80, True, {}, "Backpropagate through server and return the client-front gradient."),
    ("LOCAL_UPDATE", 90, True, {}, "Apply the configured optimizer to each model segment."),
    ("FEDERATED_AGGREGATION", 100, True, {}, "Aggregate permitted model updates and redistribute the global model."),
    ("RDP_ACCOUNT", 110, False, {}, "Record Gaussian releases with the independent conservative accountant."),
]
OPERATIONS = {row[0]: dict(zip(["op", "stage", "required", "params", "description"], row)) for row in _OP_DATA}
_ALIASES = {"CLIENT_BACKWARD": "BACKWARD", "FEDAVG_WEIGHTED": "FEDERATED_AGGREGATION", "ADD_GAUSSIAN_NOISE": "ACTIVATION_GAUSSIAN", "ADD_GRADIENT_NOISE": "GRADIENT_GAUSSIAN"}


def normalize_expression(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise FormulaError("Enter a nonempty numerical expression.")
    if len(expression) > LIMITS["expression_characters"]:
        raise FormulaError("Expression exceeds the 2000-character limit.")
    # Translate superscripts before NFKC erases their exponent semantics.
    expression = expression.replace("²", "**2").replace("³", "**3")
    expression = unicodedata.normalize("NFKC", expression)
    replacements = {"×": "*", "÷": "/", "−": "-", "^": "**", "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "η": "eta", "λ": "lambda_", "μ": "mu", "σ": "sigma", "ρ": "rho", "π": "pi", "√": "sqrt"}
    for symbol, text in replacements.items():
        expression = expression.replace(symbol, text)
    if any(symbol in expression for symbol in ("∇", "∂", "∫", "Σ", "∑", "Π", "∏", "∞", "≤", "≥", "≠", "ℙ", "∥")):
        raise FormulaError("Symbolic calculus, inequalities and distribution notation are display-only. Use executable calls such as sum(x), norm(x), gaussian(x, sigma, C), or the engine's existing gradient hook.")
    return expression.strip()


@lru_cache(maxsize=128)
def _parse(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise FormulaError("Invalid expression syntax; enter one expression, without assignments or source code.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > LIMITS["ast_nodes"]:
        raise FormulaError("Expression exceeds the 256-node complexity limit.")
    if sum(isinstance(n, (ast.Call, ast.BinOp, ast.UnaryOp)) for n in nodes) > LIMITS["operations"]:
        raise FormulaError("Expression exceeds the 80-operation limit.")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)

    def inspect(node: ast.AST, depth: int = 0) -> None:
        if depth > LIMITS["depth"]:
            raise FormulaError("Expression exceeds the 24-level nesting limit.")
        if not isinstance(node, allowed):
            raise FormulaError(f"Unsupported syntax: {type(node).__name__}. Attributes, subscripts, source code and comprehensions are forbidden.")
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float) or not math.isfinite(node.value) or abs(node.value) > 1e12):
            raise FormulaError("Only finite real numeric literals with magnitude at most 1e12 are permitted.")
        if isinstance(node, ast.Name) and (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.id) or "__" in node.id):
            raise FormulaError("Identifiers must be plain mathematical names; private or dunder names are forbidden.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_ARITY:
                raise FormulaError("Unknown operation. Only the listed numerical functions are executable; symbolic differentiation and integration are not supported.")
            low, high = _FUNCTION_ARITY[node.func.id]
            if node.keywords or not low <= len(node.args) <= high:
                raise FormulaError(f"{node.func.id} requires {low} positional argument(s); keyword arguments are forbidden.")
        for child in ast.iter_child_nodes(node):
            inspect(child, depth + 1)

    inspect(tree)
    return tree


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise FormulaError(f"{name} must be a scalar.")
        value = value.detach().item()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise FormulaError(f"{name} must be a finite real scalar.")
    return float(value)


def _finite(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
            raise FormulaError("An operation produced nonfinite or non-real tensor values.")
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise FormulaError("An operation produced a nonfinite or non-real value.")
    return value


def clip_tensor(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    """Clip the whole transmitted tensor to an L2 ball, preserving autograd."""
    if not isinstance(tensor, torch.Tensor):
        raise FormulaError("clip requires a tensor as its first argument.")
    bound = _scalar(bound, "Clipping bound")
    if bound <= 0:
        raise FormulaError("Clipping bounds must be strictly positive.")
    length = torch.linalg.vector_norm(tensor)
    return tensor * torch.clamp(bound / length.clamp_min(torch.finfo(tensor.dtype).tiny), max=1.0)


def _call(name: str, args: list[Any], generator: torch.Generator | None) -> Any:
    x = args[0]
    if name == "clip":
        return clip_tensor(x, args[1])
    if name in {"gaussian", "noise", "laplace"}:
        if not isinstance(x, torch.Tensor):
            raise FormulaError(f"{name} requires a tensor as its first argument.")
        scale = _scalar(args[1], "Noise scale")
        if scale < 0:
            raise FormulaError("Noise scale cannot be negative.")
        if name == "gaussian":
            bound = _scalar(args[2], "Gaussian clipping bound")
            if bound <= 0:
                raise FormulaError("The Gaussian clipping bound must be positive.")
            scale *= bound
        if generator is None:
            raise FormulaError("Stochastic operations require an explicit seeded torch.Generator.")
        if name == "laplace":
            u = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator) - 0.5
            # Protect the endpoint without changing the distribution on its support.
            u = u.clamp(-0.5 + torch.finfo(x.dtype).eps, 0.5 - torch.finfo(x.dtype).eps)
            draw = -u.sign() * torch.log1p(-2 * u.abs())
        else:
            draw = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        return x + draw * scale
    if name in {"min", "max"}:
        y = args[1]
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            exemplar = x if isinstance(x, torch.Tensor) else y
            x = torch.as_tensor(x, device=exemplar.device, dtype=exemplar.dtype)
            y = torch.as_tensor(y, device=exemplar.device, dtype=exemplar.dtype)
            return torch.minimum(x, y) if name == "min" else torch.maximum(x, y)
        return min(x, y) if name == "min" else max(x, y)
    if name == "clamp":
        low, high = _scalar(args[1], "Lower bound"), _scalar(args[2], "Upper bound")
        if low > high:
            raise FormulaError("clamp lower bound exceeds its upper bound.")
        return x.clamp(low, high) if isinstance(x, torch.Tensor) else min(max(x, low), high)
    if name == "dot":
        if not all(isinstance(a, torch.Tensor) for a in args) or args[0].shape != args[1].shape:
            raise FormulaError("dot requires two tensors of identical shape.")
        return (args[0] * args[1]).sum()
    if name == "normalize":
        if not isinstance(x, torch.Tensor) or x.ndim != 1 or bool((x < 0).any()) or x.sum().item() <= 0:
            raise FormulaError("normalize requires a nonnegative vector with positive sum.")
        return x / x.sum()
    if name == "softmax":
        if not isinstance(x, torch.Tensor) or x.ndim != 1:
            raise FormulaError("softmax requires a one-dimensional tensor.")
        return torch.softmax(x, dim=0)
    if name in {"sqrt", "log"}:
        bad = (x < 0) if name == "sqrt" else (x <= 0)
        if bool(bad.any()) if isinstance(bad, torch.Tensor) else bad:
            raise FormulaError(f"{name} received a value outside its real domain.")
    if isinstance(x, torch.Tensor):
        unary = {"abs": torch.abs, "norm": torch.linalg.vector_norm, "sqrt": torch.sqrt, "log": torch.log,
                 "exp": torch.exp, "sum": torch.sum, "mean": torch.mean, "var": lambda t: torch.var(t, unbiased=False),
                 "relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}
        return unary[name](x)
    unary_scalar = {"abs": abs, "norm": abs, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                    "sum": lambda a: a, "mean": lambda a: a, "var": lambda a: 0.0,
                    "relu": lambda a: max(a, 0), "sigmoid": lambda a: 1 / (1 + math.exp(-a)), "tanh": math.tanh}
    return unary_scalar[name](x)


def _evaluate(node: ast.AST, values: dict[str, Any], generator: torch.Generator | None) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values, generator)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise FormulaError(f"Missing value for {node.id} in this hook context.")
        return _finite(values[node.id])
    if isinstance(node, ast.UnaryOp):
        x = _evaluate(node.operand, values, generator)
        return _finite(-x if isinstance(node.op, ast.USub) else x)
    if isinstance(node, ast.Call):
        return _finite(_call(node.func.id, [_evaluate(arg, values, generator) for arg in node.args], generator))
    if isinstance(node, ast.BinOp):
        x, y = _evaluate(node.left, values, generator), _evaluate(node.right, values, generator)
        if isinstance(node.op, ast.Pow):
            exponent = _scalar(y, "Exponent")
            if abs(exponent) > 16:
                raise FormulaError("Exponent magnitude cannot exceed 16.")
            answer = x ** exponent
        elif isinstance(node.op, ast.Add):
            answer = x + y
        elif isinstance(node.op, ast.Sub):
            answer = x - y
        elif isinstance(node.op, ast.Mult):
            answer = x * y
        elif isinstance(node.op, ast.Div):
            zero = y == 0
            if bool(zero.any()) if isinstance(zero, torch.Tensor) else zero:
                raise FormulaError("Division by zero; guard zero norms with max(norm(x), small_positive_value).")
            answer = x / y
        else:
            raise FormulaError("Unsupported arithmetic operation.")
        return _finite(answer)
    raise FormulaError("Unsupported expression node.")


def _check_parameters(parameters: dict[str, float]) -> None:
    if len(parameters) > 40:
        raise FormulaError("At most 40 scalar parameters are supported.")
    for name, value in parameters.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or "__" in name:
            raise FormulaError(f"Invalid parameter name: {name}.")
        if name in _RESERVED:
            raise FormulaError(f"Parameter {name} is an immutable hook-context variable or operation and cannot be overridden.")
        value = _scalar(value, name)
        if abs(value) > 1e12:
            raise FormulaError(f"Parameter {name} exceeds the magnitude limit of 1e12.")
        if (name in {"C_A", "C_G", "C", "lr", "learning_rate"} or name.startswith("clip_")) and value <= 0:
            raise FormulaError(f"{name} must be strictly positive.")
        if (name.startswith("sigma") or name in {"mu", "noise_scale", "lambda_"}) and value < 0:
            raise FormulaError(f"{name} must be nonnegative.")
        if name.endswith("delta") and not 0 < value < 1:
            raise FormulaError(f"{name} must be strictly between 0 and 1.")


def _check_variables(tree: ast.Expression, hook: str, parameters: dict[str, float]) -> set[str]:
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(_FUNCTION_ARITY)
    allowed = _CONTEXT[hook] | set(parameters) | {"pi", "e"}
    unknown = names - allowed
    if unknown:
        raise FormulaError(f"Unknown or incompatible variable(s) for {hook}: {', '.join(sorted(unknown))}. Allowed context: {', '.join(sorted(_CONTEXT[hook]))}; declare other scalar parameters explicitly.")
    return names


def _check_result(hook: str, result: Any, context: dict[str, Any]) -> Any:
    _finite(result)
    source = {"H1": "A", "H2": "A", "H3": "Z", "H5": "G", "H6": "G", "H8": "weights", "H9": "rho", "H10": "score", "H13": "scores"}.get(hook)
    if source:
        expected = context.get(source)
        if expected is None and hook == "H8":
            expected = context["n"]
        if isinstance(expected, torch.Tensor):
            if not isinstance(result, torch.Tensor) or result.shape != expected.shape:
                raise FormulaError(f"{hook} output must preserve {source} tensor shape {tuple(expected.shape)}.")
            if result.device != expected.device or result.dtype != expected.dtype:
                raise FormulaError(f"{hook} output must preserve tensor dtype and device.")
        elif isinstance(result, torch.Tensor) and result.numel() != 1:
            raise FormulaError(f"{hook} requires a scalar score in this context.")
    if hook == "H8":
        if not isinstance(result, torch.Tensor) or result.ndim != 1 or bool((result < 0).any()) or not math.isclose(result.sum().item(), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise FormulaError("Aggregation weights must be a nonnegative vector summing to one.")
    if hook in {"H4", "H7", "H11", "H12", "H14"}:
        value = _scalar(result, f"{hook} output")
        if hook == "H7" and value <= 0:
            raise FormulaError("The learning rate must be strictly positive.")
        if hook in {"H11", "H12", "H14"} and value < 0:
            raise FormulaError("Time, energy and regularization outputs must be nonnegative.")
        if hook in {"H4", "H14"}:
            reference = context.get("loss" if hook == "H4" else "w")
            if isinstance(reference, torch.Tensor) and reference.requires_grad:
                if not isinstance(result, torch.Tensor) or not result.requires_grad:
                    raise FormulaError("Loss and regularizer hooks must preserve a differentiable connection to their input tensors.")
    if hook == "H9":
        if isinstance(result, torch.Tensor):
            if bool((result < 0).any()):
                raise FormulaError("Privacy-accounting diagnostics must be nonnegative.")
        elif result < 0:
            raise FormulaError("Privacy-accounting diagnostics must be nonnegative.")
    return result


def _check_context(hook: str, context: dict[str, Any]) -> None:
    """Check physical domains even when an expression masks invalid inputs."""
    positive = {"H7": ("eta",), "H8": ("N",), "H11": ("R",)}.get(hook, ())
    nonnegative = {"H8": ("n",), "H9": ("rho",), "H11": ("S", "d"),
                   "H12": ("energy", "E", "E_tx", "E_rx", "E_compute")}.get(hook, ())
    for name in positive:
        if name in context and _scalar(context[name], name) <= 0:
            raise FormulaError(f"{name} must be strictly positive in {hook}.")
    for name in nonnegative:
        if name not in context:
            continue
        value = _finite(context[name])
        if bool((value < 0).any()) if isinstance(value, torch.Tensor) else value < 0:
            raise FormulaError(f"{name} must be nonnegative in {hook}.")
    if hook == "H9":
        if "alpha" in context:
            alpha = _finite(context["alpha"])
            if bool((alpha <= 1).any()) if isinstance(alpha, torch.Tensor) else alpha <= 1:
                raise FormulaError("RDP alpha must be greater than one.")
        if "delta" in context and not 0 < _scalar(context["delta"], "delta") < 1:
            raise FormulaError("Privacy delta must be strictly between zero and one.")
    if hook == "H11" and "p" in context and not 0 <= _scalar(context["p"], "p") < 1:
        raise FormulaError("Packet loss p must be in [0, 1).")


def _check_probe_gradient(hook: str, result: Any, context: dict[str, Any]) -> None:
    """Detect disconnected and nonfinite derivatives on representative probes."""
    source = {"H1": "A", "H2": "A", "H3": "Z", "H4": "loss", "H14": "w"}.get(hook)
    if source is None or result is None:
        return
    if not isinstance(result, torch.Tensor) or not result.requires_grad:
        raise FormulaError(f"{hook} must preserve a differentiable connection to {source}.")
    gradient = torch.autograd.grad(result.sum(), context[source], allow_unused=True)[0]
    if gradient is None:
        raise FormulaError(f"{hook} has no differentiable connection to {source}.")
    if not bool(torch.isfinite(gradient).all()):
        raise FormulaError(f"{hook} produces a nonfinite derivative on a validation probe.")


def _algorithm(candidate: CandidateConfig) -> list[tuple[str, dict[str, float]]]:
    if not candidate.algorithm:
        raise FormulaError("Algorithm mode requires the controlled mandatory SFL stages.")
    if len(candidate.algorithm) > len(OPERATIONS):
        raise FormulaError("The algorithm has too many stages.")
    normalized = []
    previous = 0
    seen: set[str] = set()
    for entry in candidate.algorithm:
        name = _ALIASES.get(entry.op, entry.op)
        if name not in OPERATIONS:
            raise FormulaError(f"Unknown algorithm operation: {entry.op}. Arbitrary code and unsupported operations cannot execute.")
        spec = OPERATIONS[name]
        if name in seen or spec["stage"] <= previous:
            raise FormulaError(f"{name} is duplicated or out of order; preserve the U-shaped forward/backward and aggregation sequence.")
        if set(entry.params) != set(spec["params"]):
            raise FormulaError(f"{name} parameters must be exactly {', '.join(spec['params']) or '(none)'}.")
        for key, value in entry.params.items():
            value = _scalar(value, key)
            if value <= 0 or value > 1e12:
                raise FormulaError(f"{name} parameter {key} must be positive and no greater than 1e12.")
        if name == "ACTIVATION_GAUSSIAN" and "ACTIVATION_CLIP" not in seen:
            raise FormulaError("ACTIVATION_GAUSSIAN requires preceding ACTIVATION_CLIP to define its noise multiplier.")
        if name == "GRADIENT_GAUSSIAN" and "GRADIENT_CLIP" not in seen:
            raise FormulaError("GRADIENT_GAUSSIAN requires preceding GRADIENT_CLIP to define its noise multiplier.")
        previous = spec["stage"]
        seen.add(name)
        normalized.append((name, dict(entry.params)))
    missing = [name for name, spec in OPERATIONS.items() if spec["required"] and name not in seen]
    if missing:
        raise FormulaError(f"Missing mandatory SFL stage(s): {', '.join(missing)}.")
    return normalized


def apply_hook(candidate: CandidateConfig, hook: str, context: dict[str, Any]) -> Any:
    """Execute the selected expression or algorithm protection; inactive → None.

    Built-in activation/gradient protections on CandidateConfig are applied by
    the engine independently, and deliberately do not run twice here.
    """
    if hook not in HOOKS:
        raise FormulaError(f"Unknown SFL hook: {hook}.")
    try:
        if candidate.mode == "algorithm":
            stages = dict(_algorithm(candidate))
            if hook not in {"H2", "H6"}:
                return None
            is_activation = hook == "H2"
            prefix, variable, bound_name, sigma_name = ("ACTIVATION", "A", "C_A", "sigma_A") if is_activation else ("GRADIENT", "G", "C_G", "sigma_G")
            if f"{prefix}_CLIP" not in stages:
                return None
            bound = stages[f"{prefix}_CLIP"][bound_name]
            result = clip_tensor(context[variable], bound)
            if f"{prefix}_GAUSSIAN" in stages:
                result = _call("gaussian", [result, stages[f"{prefix}_GAUSSIAN"][sigma_name], bound], context.get("generator"))
            return _check_result(hook, result, context)
        if candidate.hook != hook:
            return None
        _check_context(hook, context)
        _check_parameters(candidate.parameters)
        tree = _parse(normalize_expression(candidate.expression))
        _check_variables(tree, hook, candidate.parameters)
        values = {**candidate.parameters, "pi": math.pi, "e": math.e}
        values.update({name: context[name] for name in _CONTEXT[hook] if name in context})
        return _check_result(hook, _evaluate(tree, values, context.get("generator")), context)
    except FormulaError:
        raise
    except (ArithmeticError, RuntimeError, TypeError, KeyError, ValueError) as exc:
        raise FormulaError(f"{hook} numerical execution failed: {exc}") from exc


def protection_spec(candidate: CandidateConfig, hook: str) -> dict[str, float] | None:
    """Return an exact clipping/noise description for recognized boundaries.

    This is deliberately conservative: arbitrary algebra does not acquire a DP
    guarantee just because it mentions clipping or Gaussian noise. Returned
    sigma is the multiplier (noise std / L2 clipping radius), not absolute std.
    Built-in CandidateConfig protections are excluded; the engine owns those.
    """
    if hook not in {"H2", "H6"}:
        return None
    if candidate.mode == "algorithm":
        operations = dict(_algorithm(candidate))
        prefix, c, s = ("ACTIVATION", "C_A", "sigma_A") if hook == "H2" else ("GRADIENT", "C_G", "sigma_G")
        if f"{prefix}_CLIP" not in operations:
            return None
        return {"clip": operations[f"{prefix}_CLIP"][c], "sigma": operations.get(f"{prefix}_GAUSSIAN", {}).get(s, 0.0)}
    if candidate.hook != hook:
        return None
    _check_parameters(candidate.parameters)
    node = _parse(normalize_expression(candidate.expression)).body
    sigma, noise_bound = 0.0, None
    if isinstance(node, ast.Call) and node.func.id == "gaussian":
        try:
            sigma = _scalar(_evaluate(node.args[1], candidate.parameters, None), "sigma")
            noise_bound = _scalar(_evaluate(node.args[2], candidate.parameters, None), "noise bound")
        except FormulaError:
            return None
        node = node.args[0]
    if not isinstance(node, ast.Call) or node.func.id != "clip":
        return None
    if not isinstance(node.args[0], ast.Name) or node.args[0].id != ("A" if hook == "H2" else "G"):
        return None
    try:
        bound = _scalar(_evaluate(node.args[1], candidate.parameters, None), "clip")
    except FormulaError:
        return None
    if bound <= 0 or sigma < 0 or (noise_bound is not None and noise_bound <= 0):
        return None
    return {"clip": bound, "sigma": sigma * noise_bound / bound if noise_bound is not None else sigma}


def _dummy_context(index: int = 0) -> dict[str, Any]:
    data = torch.tensor([[3.0, -4.0, 0.0], [-1.0, 2.0, 0.5]])
    if index == 1:
        data = torch.zeros_like(data)
    elif index == 2:
        data = data * 100
    w = data.flatten().clone().requires_grad_(True)
    n = torch.tensor([2.0, 3.0, 5.0]) if index != 1 else torch.tensor([1.0, 1.0, 1.0])
    return {"A": data.clone().requires_grad_(True), "Z": data.clone().requires_grad_(True), "G": data,
            "loss": torch.tensor(1.2 if index != 1 else 0.0, requires_grad=True), "eta": 0.01,
            "weights": n / n.sum(), "n": n, "N": float(n.sum()), "rho": torch.tensor([1.0, 2.0, 3.0]),
            "alpha": torch.tensor([2.0, 4.0, 8.0]), "delta": 1e-5,
            "score": torch.tensor([1.0, 2.0, 4.0]), "S": 10000.0, "R": 1e6,
            "p": 0.1 if index != 1 else 0.0, "d": 0.01, "energy": 2.0, "E": 2.0,
            "E_tx": 0.4, "E_rx": 0.6, "E_compute": 1.0, "C": 2.0, "latency": 0.1,
            "scores": torch.tensor([0.2, 0.8, 0.4]), "regularizer": w.square().sum() * 0,
            "w": w, "w_global": torch.zeros_like(w), "generator": torch.Generator().manual_seed(934 + index)}


def validate_candidate(candidate: CandidateConfig) -> dict[str, Any]:
    names = ["Syntax", "Variables", "Domains", "Tensor Dimensions", "SFL Hook Compatibility", "Mathematical Constraints"]
    checks = [{"name": name, "status": "pending", "detail": "Not checked."} for name in names]
    normalized = ""
    hook = HOOKS.get(candidate.hook)
    info = {"name": candidate.name, "category": candidate.mode, "input": hook["input"] if hook else None,
            "output": hook["output"] if hook else None, "parameters": candidate.parameters,
            "expected_property": "Finite values and preservation of the selected hook's mathematical contract.",
            "hook": candidate.hook, "description": hook["description"] if hook else "Unknown hook",
            "mechanisms": {}, "limitations": ["Finite tensor probes are not a general symbolic proof; all real hook outputs are checked again at runtime."]}

    def passed(index: int, detail: str) -> None:
        checks[index].update(status="pass", detail=detail)

    stage = 0
    try:
        if not hook:
            stage = 4
            raise FormulaError(f"Unknown SFL hook: {candidate.hook}.")
        if candidate.mode == "algorithm":
            sequence = _algorithm(candidate)
            normalized = " → ".join(name for name, _ in sequence)
            passed(0, "Only enumerated, typed SFL operations are present.")
            passed(1, "Operation parameters are explicitly declared; no arbitrary code.")
            passed(2, "Clipping bounds and noise multipliers are positive finite values.")
            passed(4, "Required forward, transmission, loss, backward, update and aggregation stages occur in order.")
        else:
            normalized = normalize_expression(candidate.expression)
            tree = _parse(normalized)
            passed(0, "Expression parsed within the bounded numerical language.")
            stage = 1
            _check_parameters(candidate.parameters)
            used = _check_variables(tree, candidate.hook, candidate.parameters)
            passed(1, "Declared variables: " + (", ".join(sorted(used)) or "constant expression"))
            stage = 2
            passed(2, "Parameter domains and finite scalar bounds hold.")
            passed(4, f"Expression uses only {candidate.hook} inputs and declared scalar parameters.")
        stage = 3
        for index in range(3):
            context = _dummy_context(index)
            hooks = ["H2", "H6"] if candidate.mode == "algorithm" else [candidate.hook]
            for key in hooks:
                result = apply_hook(candidate, key, context)
                _check_probe_gradient(key, result, context)
                spec = protection_spec(candidate, key)
                if spec and spec["sigma"] == 0 and result is not None:
                    if torch.linalg.vector_norm(result).item() > spec["clip"] * (1 + 1e-5):
                        raise FormulaError("The claimed clipping bound does not hold on the validation probe.")
            # Probe hooks with real scalar contexts as well as vector contexts.
            if candidate.mode == "formula" and candidate.hook in {"H9", "H10"}:
                context.update(rho=1.0, alpha=4.0, score=1.0)
                apply_hook(candidate, candidate.hook, context)
        passed(3, "Representative real tensors, zero tensors and large-magnitude tensors satisfy output shapes, finiteness, finite derivatives and scalar requirements.")
        passed(5, "Runtime guards enforce positive learning rates/bounds, legal noise scales, finite real domains and normalized nonnegative aggregation weights where applicable.")
        for key in ("H2", "H6"):
            spec = protection_spec(candidate, key)
            if spec:
                info["mechanisms"][key] = spec
        if candidate.hook == "H9":
            info["limitations"].append("H9 is an exploratory accounting diagnostic and cannot weaken or replace the independent conservative privacy guarantee.")
        if candidate.mode == "algorithm":
            info["algorithm"] = [{"op": name, "params": params} for name, params in sequence]
    except (FormulaError, ArithmeticError, RuntimeError, ValueError, TypeError) as exc:
        message = str(exc)
        # Numerical-domain failures deserve a domain label, not a shape label.
        if stage == 3 and any(part in message.lower() for part in ("domain", "division by zero", "nonfinite", "positive", "negative", "exponent")):
            stage = 2
        if stage == 3 and any(part in message.lower() for part in ("aggregation weights", "clipping bound", "differentiable")):
            stage = 5
        checks[stage].update(status="fail", detail=message)
        for check in checks:
            if check["status"] == "pending":
                check.update(status="skipped", detail="Requires earlier validation to pass.")
        return {"valid": False, "checks": checks, "normalized_expression": normalized, "expression": normalized, "formula_info": info, "error": message}
    return {"valid": True, "checks": checks, "normalized_expression": normalized, "expression": normalized, "formula_info": info, "error": None}


def _key(label: str, insert: str, description: str, executable: bool = True) -> dict[str, Any]:
    return {"label": label, "insert": insert, "description": description, "executable": executable}


def catalog() -> dict[str, Any]:
    templates = [
        {"id": "gradient_clip", "name": "Gradient clipping", "hook": "H6", "expression": "clip(G, C_G)", "parameters": {"C_G": 1.0}, "expected_property": "Whole transmitted gradient has L2 norm at most C_G."},
        {"id": "activation_clip", "name": "Activation clipping", "hook": "H2", "expression": "clip(A, C_A)", "parameters": {"C_A": 1.0}, "expected_property": "Whole transmitted activation has L2 norm at most C_A."},
        {"id": "gradient_gaussian", "name": "Clipped Gaussian gradient", "hook": "H6", "expression": "gaussian(clip(G, C_G), sigma_G, C_G)", "parameters": {"C_G": 1.0, "sigma_G": 1.0}, "expected_property": "Gaussian standard deviation = sigma_G × C_G; bounded tensor transcript mechanism."},
        {"id": "activation_gaussian", "name": "Clipped Gaussian activation", "hook": "H2", "expression": "gaussian(clip(A, C_A), sigma_A, C_A)", "parameters": {"C_A": 1.0, "sigma_A": 1.0}, "expected_property": "Gaussian standard deviation = sigma_A × C_A; bounded tensor transcript mechanism."},
        {"id": "fedavg", "name": "Weighted FedAvg", "hook": "H8", "expression": "n / N", "parameters": {}, "expected_property": "Nonnegative sample weights sum to one over participating clients."},
        {"id": "equal_weights", "name": "Equal averaging", "hook": "H8", "expression": "softmax(n * 0)", "parameters": {}, "expected_property": "Each participating client has equal aggregation weight."},
        {"id": "fedprox", "name": "FedProx regularizer", "hook": "H14", "expression": "(mu / 2) * norm(w - w_global)**2", "parameters": {"mu": 0.01}, "expected_property": "Nonnegative differentiable penalty toward the round's global parameters."},
        {"id": "communication", "name": "Communication with retransmissions", "hook": "H11", "expression": "S / (R * (1 - p)) + d", "parameters": {}, "expected_property": "Estimated seconds from transmitted bits and available bit rate."},
        {"id": "energy", "name": "Client energy components", "hook": "H12", "expression": "E_compute + E_tx + E_rx", "parameters": {}, "expected_property": "Nonnegative energy estimate in joules."},
        {"id": "rdp_composition", "name": "RDP composition diagnostic", "hook": "H9", "expression": "rho * steps", "parameters": {"steps": 2.0}, "expected_property": "Exploratory composition; certified accounting is calculated independently."},
        {"id": "rdp_conversion", "name": "RDP conversion diagnostic", "hook": "H9", "expression": "rho + log(1 / delta) / (alpha - 1)", "parameters": {}, "expected_property": "Per-order epsilon bound diagnostic; the engine's certified accountant minimizes across orders."},
        {"id": "learning_rate", "name": "Learning-rate multiplier", "hook": "H7", "expression": "eta * factor", "parameters": {"factor": 0.5}, "expected_property": "Positive candidate learning rate."},
        {"id": "client_forward", "name": "Bounded client activations", "hook": "H1", "expression": "tanh(A)", "parameters": {}, "expected_property": "Differentiable elementwise activations in [-1, 1]."},
        {"id": "server_forward", "name": "Server activation scaling", "hook": "H3", "expression": "Z * factor", "parameters": {"factor": 0.9}, "expected_property": "Preserves server tensor dimensions and gradient connectivity."},
        {"id": "loss_scale", "name": "Loss scaling", "hook": "H4", "expression": "loss * factor", "parameters": {"factor": 1.0}, "expected_property": "Differentiable scaling of the local classification objective."},
        {"id": "gradient_scale", "name": "Gradient scaling", "hook": "H5", "expression": "G * factor", "parameters": {"factor": 0.9}, "expected_property": "Preserves outgoing gradient dimensions."},
        {"id": "split_score", "name": "Measured communication split score", "hook": "H10", "expression": "score", "parameters": {}, "expected_property": "Selects the feasible split with the lowest measured boundary payload score."},
        {"id": "client_selection", "name": "Sample-count client selection", "hook": "H13", "expression": "n", "parameters": {}, "expected_property": "Prioritizes clients with more local training examples; participation count is unchanged."},
    ]
    keyboard = {
        "Basic": [_key(s, ins, desc, executable) for s, ins, desc, executable in [
            ("+", "+", "Addition", True), ("−", "-", "Subtraction", True), ("×", "*", "Multiplication", True), ("÷", "/", "Division", True),
            ("=", "=", "Equality notation; assignments are display-only", False), ("<", "<", "Comparison notation is display-only", False), (">", ">", "Comparison notation is display-only", False),
            ("≤", "≤", "Less than or equal; display-only", False), ("≥", "≥", "Greater than or equal; display-only", False),
            ("(", "(", "Open parenthesis", True), (")", ")", "Close parenthesis", True), ("[ ]", "[]", "Bracket notation is display-only", False), ("{ }", "{}", "Set notation is display-only", False),
            ("x²", "**2", "Square the preceding expression", True), ("xⁿ", "**", "Power; scalar exponent magnitude at most 16", True), ("√x", "sqrt(", "Real square root", True),
            ("1/x", "1 / (", "Reciprocal", True), ("|x|", "abs(", "Elementwise absolute value", True), ("min", "min(", "Elementwise minimum of two arguments", True), ("max", "max(", "Elementwise maximum of two arguments", True)]],
        "Greek": [_key(symbol, name, description) for symbol, name, description in [
            ("α", "alpha", "RDP order in H9; otherwise declare a scalar parameter"), ("β", "beta", "User-defined beta parameter"), ("γ", "gamma", "User-defined gamma parameter"),
            ("δ", "delta", "Approximate-DP failure probability in H9"), ("ε", "epsilon", "DP budget; lower generally means stronger privacy under fixed assumptions"),
            ("η", "eta", "Learning rate in H7"), ("λ", "lambda_", "Regularization coefficient"), ("μ", "mu", "FedProx strength"), ("σ", "sigma", "Noise multiplier or scale as defined by the chosen operation"), ("ρ", "rho", "RDP privacy cost in H9"), ("π", "pi", "Mathematical constant pi")]],
        "Calculus": [_key("Σ", "sum(", "Sum all elements of an available tensor"), _key("Π", "Π", "Symbolic product is display-only", False), _key("∂", "∂", "Partial derivative is display-only; engine computes gradients", False), _key("∇", "∇", "Gradient notation is display-only; use H5/H6", False), _key("∫", "∫", "Integration is display-only", False), _key("lim", "lim(", "Symbolic limit is not executable", False)],
        "Probability": [_key("E[·]", "mean(", "Empirical tensor mean"), _key("Var[·]", "var(", "Population variance of tensor entries"), _key("P(·)", "P(", "Symbolic probability is display-only", False), _key("log", "log(", "Natural logarithm with positive input"), _key("exp", "exp(", "Natural exponential"), _key("N(0,σ²I)", "gaussian(", "gaussian(tensor, multiplier, clip_bound), standard deviation multiplier × bound"), _key("Laplace", "laplace(", "laplace(tensor, absolute_scale); no certified Laplace accountant")],
        "Differential Privacy": [_key("clip", "clip(", "Whole-tensor L2 clipping: clip(tensor, positive_bound)"), _key("Gaussian", "gaussian(", "Add Gaussian noise calibrated to clipping bound"), _key("ε", "epsilon", "DP epsilon; explicitly declare scalar parameters outside accounting contexts"), _key("δ", "delta", "DP failure probability in the accounting context"), _key("ρ_total", "rho", "Accumulated RDP by order; diagnostics cannot replace certified accounting"), _key("α", "alpha", "Rényi order, greater than one")],
        "SFL": [_key(label, insert, desc) for label, insert, desc in [
            ("Aₖ", "A", "Client-front activation, H1/H2"), ("Zₖ", "Z", "Server output, H3"), ("Gₖ", "G", "Split gradient, H5/H6"),
            ("Wₖ", "w", "Local flattened parameters, H14"), ("Wₜ", "w_global", "Round-global flattened parameters, H14"), ("W_G", "w_global", "Global flattened parameters, H14"),
            ("nₖ", "n", "Client sample counts, H8/H13"), ("N", "N", "Total participating samples, H8"), ("K", "K", "Client count; declare explicitly when needed"),
            ("C_A", "C_A", "Activation L2 clipping bound"), ("C_G", "C_G", "Gradient L2 clipping bound"), ("σ_A", "sigma_A", "Activation Gaussian noise multiplier"), ("σ_G", "sigma_G", "Gradient Gaussian noise multiplier"),
            ("ε", "epsilon", "DP budget under the specified accountant"), ("δ", "delta", "Approximate-DP failure probability"), ("ε_total", "epsilon_total", "Total epsilon; available only if declared"), ("η", "eta", "Optimizer learning rate, H7"), ("μ", "mu", "FedProx coefficient"),
            ("T_comm", "T_comm", "Communication seconds; declare parameter if needed"), ("T_total", "T_total", "Total seconds; declare parameter if needed"), ("E_client", "energy", "Client energy estimate, H12"), ("E_tx", "E_tx", "Transmit energy estimate, H12"), ("E_rx", "E_rx", "Receive energy estimate, H12"),
            ("Accuracy", "Accuracy", "Measured accuracy; not available during training to prevent leakage"), ("Loss", "loss", "Client-local differentiable loss, H4"), ("Drift", "Drift", "Local/global parameter distance; declare explicitly"), ("Bandwidth", "R", "Available bit rate, H11"), ("PacketLoss", "p", "Packet loss probability, H11")]],
        "Norms": [_key("‖x‖₂", "norm(", "Whole-tensor L2 norm"), _key("normalize", "normalize(", "Normalize a nonnegative vector to sum to one"), _key("dot", "dot(", "Inner product of two identically shaped tensors"), _key("clamp", "clamp(", "Elementwise clamp to scalar lower and upper bounds")],
        "Optimization": [_key("η", "eta", "Positive optimizer learning rate"), _key("μ", "mu", "Nonnegative FedProx coefficient"), _key("ReLU", "relu(", "Elementwise positive part"), _key("sigmoid", "sigmoid(", "Logistic activation"), _key("softmax", "softmax(", "Convert a vector of scores into normalized weights"), _key("tanh", "tanh(", "Hyperbolic tangent")],
        "Communication": [_key("S", "S", "Transmitted bits in H11"), _key("R", "R", "Bandwidth in bits per second"), _key("p", "p", "Packet loss probability in [0,1)"), _key("d", "d", "Additional delay in seconds"), _key("E", "energy", "Total estimated joules"), _key("E_compute", "E_compute", "Estimated computation energy")],
    }
    return {"hooks": list(HOOKS.values()), "templates": templates, "operations": list(OPERATIONS.values()), "keyboard": keyboard,
            "default_algorithm": [{"op": name, "params": {}} for name, spec in OPERATIONS.items() if spec["required"]],
            "functions": list(_FUNCTION_ARITY), "limits": LIMITS,
            "notes": ["Expression syntax is a bounded numerical language, never arbitrary Python.", "Keyboard notation marked display-only is rejected explicitly by validation.", "Gaussian sigma is a multiplier: standard deviation = sigma × clipping bound.", "H9 is a diagnostic; privacy certification comes from the independent conservative accountant."]}
