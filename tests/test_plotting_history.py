"""amigo.plotting.plot draws both the legacy Trainer's ParamHistory and a new FitResult."""

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax
import zodiax as zdx

import amigo.plotting
from amigo.optimisation import Trainer


class Toy(zdx.Base):
    positions: dict
    fluxes: dict
    rotation: jax.Array

    def __init__(self):
        self.positions = {"a": np.array([1.0, 2.0]), "b": np.array([0.0, -1.0])}
        self.fluxes = {"a": np.array(3.0), "b": np.array(4.0)}
        self.rotation = np.array(0.5)


def loss(model, args):
    leaves = jax.tree_util.tree_leaves((model.positions, model.fluxes, model.rotation))
    return sum(np.sum(leaf**2) for leaf in leaves)


def train_toy(epochs=3):
    optimisers = {name: optax.sgd(0.1) for name in ("positions", "fluxes", "rotation")}
    return Trainer(loss_fn=loss).train(Toy(), optimisers, epochs=epochs)


def count_figures(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(1))
    return shown


def test_fit_result_regroups_paths():
    history = amigo.plotting._history(train_toy(epochs=3))
    params = history.params
    assert set(params) == {"positions", "fluxes", "rotation"}
    assert set(params["positions"]) == {"a", "b"}
    assert len(params["positions"]["a"]) == 4  # epoch 0 plus three updates
    assert len(params["rotation"]) == 4  # a parameter without a key stays a list


def test_plot_fit_result_and_its_history(monkeypatch):
    result = train_toy()
    shown = count_figures(monkeypatch)
    amigo.plotting.plot(result)
    amigo.plotting.plot(result.parameter_history)
    assert len(shown) == 4  # three parameters, two per figure, twice
    plt.close("all")


def test_plot_legacy_history_unchanged(monkeypatch):
    legacy = SimpleNamespace(
        params={
            "positions": {"a": [np.array([1.0, 2.0]), np.array([1.1, 2.1])]},
            "fluxes": {"a": [np.array(3.0), np.array(3.1)]},
        }
    )
    assert amigo.plotting._history(legacy) is legacy
    shown = count_figures(monkeypatch)
    amigo.plotting.plot(legacy)
    assert len(shown) == 1
    plt.close("all")
