"""Objective arrays must cross compilation boundaries as runtime operands."""

import unittest
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy.testing as npt
import optax
import zodiax as zdx
from amigo.optimisation import Trainer, _coordinate_loss, _value_gradient


class Model(zdx.Base):
    value: jax.Array


def residual(model, data):
    return (model.value - data) / np.sqrt(data.size)


class DynamicObjectiveTests(unittest.TestCase):
    def test_data_are_operands_and_changing_them_changes_gradient(self):
        model = Model(np.array(0.0))
        data = np.ones(8192)
        trainer = Trainer(residual_fn=residual, args=data).normalise(
            model, ["value"], method="none"
        )
        _, coordinates = trainer._coordinates(model, None)
        loss, _ = trainer._objective()
        objective = eqx.Partial(_coordinate_loss, loss, coordinates)
        closed, _, _ = eqx.filter_make_jaxpr(lambda z, fn: _value_gradient(z, fn))(
            coordinates.zero, objective
        )
        self.assertFalse(any(getattr(x, "size", 0) >= data.size for x in closed.consts))
        self.assertTrue(
            any(getattr(x.aval, "shape", ()) == data.shape for x in closed.jaxpr.invars)
        )
        npt.assert_allclose(_value_gradient(coordinates.zero, objective)[1], [-1.0])
        loss2, _ = Trainer(residual_fn=residual, args=data * 2)._objective()
        npt.assert_allclose(
            _value_gradient(coordinates.zero, eqx.Partial(_coordinate_loss, loss2, coordinates))[
                1
            ],
            [-2.0],
        )

    def test_all_optimisers_and_normalisation_use_explicit_data(self):
        model = Model(np.array(0.0))
        trainer = Trainer(residual_fn=residual, args=np.full(32, 3.0)).normalise(model, ["value"])
        for method in (trainer.gd, trainer.bfgs, trainer.lm):
            fit = method(model, max_steps=40, tolerance=1e-7)
            self.assertTrue(fit.accepted, fit.status)
            npt.assert_allclose(fit.model.value, 3.0, atol=1e-7)
        trained = trainer.train(model, {"value": optax.sgd(1.0)}, epochs=3, tolerance=1e-7)
        self.assertTrue(trained.accepted, trained.status)
        npt.assert_allclose(trained.model.value, 3.0, atol=1e-7)

    def test_custom_loss_aux_and_hessian(self):
        def loss(model, data):
            value = 0.5 * np.sum(residual(model, data) ** 2)
            return value, {"target": data.mean()}

        model = Model(np.array(0.0))
        trainer = Trainer(loss_fn=loss, residual_fn=residual, args=np.full(8, 2.0)).normalise(
            model, ["value"], method="full"
        )
        result = trainer.lm(model, max_steps=30, tolerance=1e-7)
        self.assertTrue(result.accepted, result.status)
        npt.assert_allclose(result.aux["target"], 2.0)

    def test_scalar_only_objective_cannot_request_residual_information(self):
        model = Model(np.array(0.0))
        trainer = Trainer(loss_fn=lambda m, args: 0.5 * m.value**2)
        with self.assertRaisesRegex(ValueError, "requires residual_fn"):
            trainer.normalise(model, ["value"], source="residual")


if __name__ == "__main__":
    unittest.main()
