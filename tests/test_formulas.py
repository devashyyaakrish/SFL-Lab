"""Executable specifications for the bounded, typed formula language."""

import ast
import inspect
import math

import pytest
import torch

from sfl_lab import formulas
from sfl_lab.config import CandidateConfig
from sfl_lab.formulas import FormulaError, apply_hook, catalog, clip_tensor, protection_spec, validate_candidate


def candidate(expression, hook="H6", **parameters):
    return CandidateConfig(hook=hook, expression=expression, parameters=parameters)


def algorithm(*optional):
    names = {stage["op"] for stage in optional}
    stages = [dict(op=stage["op"], params=stage["params"].copy())
              for stage in catalog()["operations"] if stage["required"] or stage["op"] in names]
    for stage in stages:
        override = next((item for item in optional if item["op"] == stage["op"]), None)
        if override:
            stage.update(override)
    return CandidateConfig(mode="algorithm", algorithm=stages)


@pytest.mark.parametrize("template", catalog()["templates"], ids=lambda template: template["id"])
def test_every_catalog_template_passes_validation(template):
    selected = CandidateConfig(**{key: template[key] for key in ("name", "hook", "expression", "parameters")})
    result = validate_candidate(selected)
    assert result["valid"], result
    assert all(check["status"] == "pass" for check in result["checks"])


def test_catalog_covers_every_hook_and_keyboard_explains_display_only_notation():
    contents = catalog()
    assert {template["hook"] for template in contents["templates"]} == {f"H{i}" for i in range(1, 15)}
    assert len(contents["hooks"]) == 14
    assert len({template["id"] for template in contents["templates"]}) == len(contents["templates"])
    assert set(contents["keyboard"]) >= {"Basic", "Greek", "Calculus", "Probability", "SFL", "Differential Privacy"}
    for keys in contents["keyboard"].values():
        for key in keys:
            assert key["description"] and isinstance(key["executable"], bool)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo unsafe')", "G.__class__", "G[0]", "G.detach()",
    "[x for x in G]", "(x for x in G)", "lambda: G", "(x := G)",
    "G if 1 else G", "G > 0", "G and G", "{1: G}", "[G]", "(G, G)",
    "f'{G}'", "'G'", "True", "None", "1j", "import os", "G; G",
    "open('file')", "eval('G')", "exec('G')", "torch.randn(3)",
    "clip(G, bound=1)", "clip(G)", "clip(G, 1, 2)", "clip(*G)",
    "G // 2", "G % 2", "G @ G", "G | G", "∇G", "∫ G", "G ≤ 1",
])
def test_source_code_and_unsupported_notation_are_rejected(expression):
    result = validate_candidate(candidate(expression))
    assert not result["valid"]
    assert result["error"]


def test_implementation_never_calls_python_eval_compile_or_exec():
    module = ast.parse(inspect.getsource(formulas))
    forbidden = {"eval", "exec", "compile", "__import__"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in forbidden for node in ast.walk(module))


@pytest.mark.parametrize("expression", ["G" * 2001, "+" * 40 + "G", "+".join(["G"] * 90), "G * 1e400", "G * 1000000000001"])
def test_complexity_and_literal_bounds(expression):
    assert not validate_candidate(candidate(expression))["valid"]


@pytest.mark.parametrize("parameters", [{"G": 1}, {"eta": 1}, {"pi": 1}, {"e": 1}, {"clip": 1},
                                         {"generator": 1}, {"__secret": 1}, {"a__b": 1}, {"a.b": 1},
                                         {"C_G": 0}, {"C_A": -1}, {"sigma_G": -1}, {"mu": -1},
                                         {"lambda_": -1}, {"privacy_delta": 1}, {"large": 1e13}])
def test_immutable_context_and_parameter_domains(parameters):
    assert not validate_candidate(CandidateConfig(expression="G", parameters=parameters))["valid"]


def test_parameter_count_limit():
    assert not validate_candidate(CandidateConfig(expression="G", parameters={f"a{i}": 1 for i in range(41)}))["valid"]


@pytest.mark.parametrize("expression,hook", [("A", "H6"), ("G", "H2"), ("loss", "H7"), ("unknown", "H8"), ("R", "H10")])
def test_cross_hook_variables_are_rejected(expression, hook):
    result = validate_candidate(candidate(expression, hook))
    assert not result["valid"] and "variable" in result["error"]


@pytest.mark.parametrize("expression", ["G / 0", "G / norm(G)", "sqrt(G)", "log(abs(G))", "G**17", "G**-17",
                                         "G + (-1)**.5", "G * exp(1000)", "G + 0**-1", "clamp(G, 2, 1)",
                                         "clip(G, -1)", "gaussian(G, -1, 1)", "gaussian(G, 1, 0)", "noise(G, -1)"])
def test_invalid_real_numerical_domains(expression):
    assert not validate_candidate(candidate(expression))["valid"]


@pytest.mark.parametrize("expression,hook", [("sum(G)", "H6"), ("1", "H2"), ("sum(score)", "H10"),
                                          ("sum(scores)", "H13"), ("-n / N", "H8"), ("n", "H8"),
                                          ("eta * 0", "H7"), ("-1", "H11"), ("-energy", "H12"),
                                          ("1", "H4"), ("sum(w_global)", "H14"), ("-rho", "H9")])
def test_hook_output_shape_and_mathematical_contracts(expression, hook):
    assert not validate_candidate(candidate(expression, hook))["valid"]


@pytest.mark.parametrize("expression,hook", [("sqrt(loss)", "H4"), ("sqrt(abs(A))", "H1"), ("sqrt(abs(Z))", "H3")])
def test_finite_forward_values_do_not_hide_nonfinite_gradients(expression, hook):
    result = validate_candidate(candidate(expression, hook))
    assert not result["valid"] and "derivative" in result["error"]


def test_clip_exact_norm_and_autograd_including_zero_input():
    source = torch.tensor([3., 4.], requires_grad=True)
    clipped = clip_tensor(source, 2.)
    torch.testing.assert_close(clipped, torch.tensor([1.2, 1.6]))
    assert clipped.norm().item() == pytest.approx(2)
    clipped.sum().backward()
    assert torch.isfinite(source.grad).all()
    zeros = torch.zeros(4, requires_grad=True)
    clip_tensor(zeros, 1).sum().backward()
    torch.testing.assert_close(zeros.grad, torch.ones(4))


@pytest.mark.parametrize("expression,expected", [
    ("abs(G)", [3., 4.]), ("relu(G)", [3., 0.]), ("G * 2 + 1", [7., -7.]),
    ("clamp(G, -2, 2)", [2., -2.]), ("min(G, 1)", [1., -4.]), ("max(G, -1)", [3., -1.]),
    ("G + dot(G, G)", [28., 21.]), ("G + norm(G)", [8., 1.]),
    ("G + var(G)", [15.25, 8.25]), ("G + mean(G)", [2.5, -4.5]),
    ("G + sum(G)", [2., -5.]), ("G + log(exp(2))", [5., -2.]),
])
def test_exact_numerical_function_values(expression, expected):
    actual = apply_hook(candidate(expression), "H6", {"G": torch.tensor([3., -4.])})
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_gaussian_multiplier_and_seed_are_exact():
    selected = candidate("gaussian(clip(G, C_G), sigma_G, C_G)", C_G=2., sigma_G=.5)
    source = torch.tensor([3., 4.])
    generator = torch.Generator().manual_seed(17)
    expected = torch.tensor([1.2, 1.6]) + torch.randn(2, generator=torch.Generator().manual_seed(17))
    actual = apply_hook(selected, "H6", {"G": source, "generator": generator})
    torch.testing.assert_close(actual, expected)
    assert protection_spec(selected, "H6") == {"clip": 2., "sigma": .5}


@pytest.mark.parametrize("expression", ["noise(G, .2)", "laplace(G, .2)", "gaussian(G, .2, 1)"])
def test_noise_requires_explicit_seed_and_is_reproducible(expression):
    selected = candidate(expression)
    with pytest.raises(FormulaError, match="seeded"):
        apply_hook(selected, "H6", {"G": torch.ones(12)})
    a = apply_hook(selected, "H6", {"G": torch.ones(12), "generator": torch.Generator().manual_seed(13)})
    b = apply_hook(selected, "H6", {"G": torch.ones(12), "generator": torch.Generator().manual_seed(13)})
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, torch.ones(12))


def test_exact_fedavg_fedprox_communication_energy_and_rdp():
    weights = apply_hook(candidate("n / N", "H8"), "H8", {"n": torch.tensor([2., 3., 5.]), "N": 10.})
    torch.testing.assert_close(weights, torch.tensor([.2, .3, .5]))
    w = torch.tensor([3., 4.], requires_grad=True)
    penalty = apply_hook(candidate("(mu / 2) * norm(w - w_global)**2", "H14", mu=.2), "H14", {"w": w, "w_global": torch.zeros(2)})
    assert penalty.item() == pytest.approx(2.5)
    penalty.backward()
    torch.testing.assert_close(w.grad, torch.tensor([.6, .8]))
    seconds = apply_hook(candidate("S / (R * (1 - p)) + d", "H11"), "H11", {"S": 80., "R": 100., "p": .2, "d": .05})
    assert seconds == pytest.approx(1.05)
    energy = apply_hook(candidate("E_compute + E_tx + E_rx", "H12"), "H12", {"E_compute": 2., "E_tx": .4, "E_rx": .6})
    assert energy == pytest.approx(3.)
    epsilon = apply_hook(candidate("rho + log(1 / delta) / (alpha - 1)", "H9"), "H9", {"rho": 2., "alpha": 4., "delta": 1e-5})
    assert epsilon == pytest.approx(2 + math.log(1e5) / 3)


@pytest.mark.parametrize("hook,expression,context", [
    ("H11", "0", {"R": 0}), ("H11", "0", {"p": 1}), ("H11", "0", {"p": -.1}),
    ("H11", "0", {"S": -1}), ("H12", "0", {"energy": -1}),
    ("H9", "rho", {"rho": 1., "alpha": 1}), ("H9", "rho", {"rho": 1., "delta": 0}),
    ("H8", "normalize(n)", {"n": torch.tensor([1., 2.]), "N": 0}),
])
def test_context_domains_are_enforced_even_if_formula_hides_invalid_value(hook, expression, context):
    with pytest.raises(FormulaError):
        apply_hook(candidate(expression, hook), hook, context)


def test_runtime_rechecks_values_that_passed_probes():
    selected = candidate("sqrt(G + 1000)")
    assert validate_candidate(selected)["valid"]
    with pytest.raises(FormulaError, match="domain"):
        apply_hook(selected, "H6", {"G": torch.tensor([-1001.])})
    selected = candidate("normalize(n)", "H8")
    with pytest.raises(FormulaError):
        apply_hook(selected, "H8", {"n": torch.zeros(3)})


def test_unicode_normalizes_executable_math():
    selected = candidate("clip(G, C_G) × σ_G + G² ÷ 4", C_G=1., sigma_G=2.)
    actual = apply_hook(selected, "H6", {"G": torch.tensor([3., 4.])})
    torch.testing.assert_close(actual, torch.tensor([3.45, 5.6]))
    assert "sigma_G" in validate_candidate(selected)["normalized_expression"]


@pytest.mark.parametrize("expression", ["gaussian(G, 1, 1)", "noise(clip(G, 1), 1)", "gaussian(clip(G, 1), 1, 1) + G", "clip(G * 2, 1)"])
def test_privacy_recognition_is_conservative(expression):
    assert protection_spec(candidate(expression), "H6") is None


def test_privacy_recognition_converts_different_noise_and_clipping_bounds():
    selected = candidate("gaussian(clip(G, bound), sigma, noise_bound)", bound=2., sigma=3., noise_bound=4.)
    assert protection_spec(selected, "H6") == {"clip": 2., "sigma": 6.}
    assert protection_spec(selected, "H2") is None
    assert protection_spec(selected, "H4") is None


def test_default_algorithm_and_all_optional_stages_validate():
    assert validate_candidate(algorithm())["valid"]
    selected = algorithm(*[{"op": stage["op"], "params": stage["params"]} for stage in catalog()["operations"] if not stage["required"]])
    result = validate_candidate(selected)
    assert result["valid"], result
    assert protection_spec(selected, "H2") == {"clip": 1., "sigma": 1.}
    assert protection_spec(selected, "H6") == {"clip": 1., "sigma": 1.}
    for hook, variable in [("H2", "A"), ("H6", "G")]:
        actual = apply_hook(selected, hook, {variable: torch.tensor([3., 4.]), "generator": torch.Generator().manual_seed(11)})
        expected = torch.tensor([.6, .8]) + torch.randn(2, generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(actual, expected)
    assert apply_hook(selected, "H4", {"loss": torch.tensor(1.)}) is None


@pytest.mark.parametrize("removed", [stage["op"] for stage in catalog()["operations"] if stage["required"]])
def test_algorithm_rejects_every_missing_mandatory_stage(removed):
    selected = algorithm()
    selected.algorithm = [entry for entry in selected.algorithm if entry.op != removed]
    assert not validate_candidate(selected)["valid"]


@pytest.mark.parametrize("operation", ["ACTIVATION_CLIP", "ACTIVATION_GAUSSIAN", "GRADIENT_CLIP", "GRADIENT_GAUSSIAN"])
def test_algorithm_rejects_protections_placed_after_aggregation(operation):
    selected = algorithm({"op": operation})
    stage = next(stage for stage in selected.algorithm if stage.op == operation)
    selected.algorithm.remove(stage)
    selected.algorithm.append(stage)
    assert not validate_candidate(selected)["valid"]


@pytest.mark.parametrize("operation", ["ACTIVATION_GAUSSIAN", "GRADIENT_GAUSSIAN"])
def test_algorithm_noise_requires_preceding_clipping(operation):
    assert not validate_candidate(algorithm({"op": operation}))["valid"]


def test_algorithm_rejects_duplicate_stages_extra_params_and_unknown_code():
    selected = algorithm()
    selected.algorithm.insert(1, selected.algorithm[0])
    assert not validate_candidate(selected)["valid"]
    selected = algorithm()
    selected.algorithm[0].params = {"unexpected": 1.}
    assert not validate_candidate(selected)["valid"]
    selected = algorithm()
    selected.algorithm[0].op = "eval('1')"
    assert not validate_candidate(selected)["valid"]


@pytest.mark.parametrize("value", [0., -1., 1e13])
def test_algorithm_parameter_domains(value):
    selected = algorithm({"op": "ACTIVATION_CLIP", "params": {"C_A": value}})
    assert not validate_candidate(selected)["valid"]


def test_inactive_hook_does_not_execute_formula_and_invalid_hook_is_rejected():
    assert apply_hook(candidate("G / 0"), "H2", {"A": torch.ones(2)}) is None
    with pytest.raises(FormulaError, match="Unknown SFL hook"):
        apply_hook(candidate("G"), "H15", {})
    assert not validate_candidate(candidate("G", "H15"))["valid"]
