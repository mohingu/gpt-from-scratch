"""Gradient checks for the micrograd autograd engine.

The engine is validated by replaying the same expression through PyTorch's autograd
and asserting both the forward values and the backward gradients agree. PyTorch is the
reference implementation; micrograd is the thing under test.
"""

import torch

from components.micrograd.engine import Value
from components.micrograd.nn import MLP


def test_sanity_check() -> None:
    """A small expression: forward value and dy/dx must match PyTorch."""
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    xt = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    z = 2 * xt + 2 + xt
    q = z.relu() + z * xt
    h = (z * z).relu()
    y = h + q + q * xt
    y.backward()
    xpt, ypt = xt, y

    # forward pass agrees
    assert ymg.data == ypt.data.item()
    # backward pass agrees
    assert xmg.grad == xpt.grad.item()


def test_more_ops() -> None:
    """A denser expression exercising add, mul, pow, div, neg, sub and relu."""
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    at = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    bt = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    c = at + bt
    d = at * bt + bt**3
    c = c + c + 1
    c = c + 1 + c + (-at)
    d = d + d * 2 + (bt + at).relu()
    d = d + 3 * d + (bt - at).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = at, bt, g

    tol = 1e-6
    # forward pass agrees
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass agrees
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_mlp_parameter_count() -> None:
    """An MLP exposes one weight per input plus one bias for every neuron."""
    model = MLP(2, [4, 1])
    # layer 1: 4 neurons x (2 weights + 1 bias) = 12; layer 2: 1 x (4 + 1) = 5
    assert len(model.parameters()) == 17


def test_mlp_forward_and_zero_grad() -> None:
    """A single scalar output, and zero_grad clears accumulated gradients."""
    model = MLP(2, [4, 1])
    out = model([Value(1.0), Value(-2.0)])
    assert isinstance(out, Value)

    out.backward()
    assert any(p.grad != 0 for p in model.parameters())

    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())
