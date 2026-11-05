"""Default batching must preserve diagonal curvature, including padded batches."""
from unittest import TestCase
from unittest.mock import patch
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as np
import numpy.testing as npt
import zodiax as zdx
from amigo import optimisation as opt


class Model(zdx.Base):
    values: jax.Array


class NormalisationBatchTests(TestCase):
    def test_default_and_explicit_batches_agree(self):
        model = Model(values=np.linspace(1., 2., 65))
        weights = np.linspace(.5, 3., 65)
        trainer = opt.Trainer(residual_fn=lambda model, args: weights * model.values)
        with patch.object(opt, '_residual_diagonal', wraps=opt._residual_diagonal) as diagonal:
            default = trainer.normalise(model, ['values'])
            self.assertEqual(diagonal.call_args.args[2:], (5, False))
        explicit = trainer.normalise(model, ['values'], nbatches=1)
        npt.assert_allclose(default.information['values'].vector, weights ** 2, rtol=1e-12)
        npt.assert_allclose(default.information['values'].vector,
                            explicit.information['values'].vector, rtol=1e-12)

    def test_linearisation_is_staged_with_reduction(self):
        model = Model(values=np.linspace(.1, 1., 33))
        trainer = opt.Trainer(residual_fn=lambda model, args: np.sin(model.values))
        original = opt._residual_diagonal
        def staged(residual, origin, nbatches, checkpoint):
            self.assertIsInstance(origin, jax.core.Tracer)
            return original(residual, origin, nbatches, checkpoint)
        with patch.object(opt, '_residual_diagonal', side_effect=staged):
            result = trainer.normalise(model, ['values'])
        npt.assert_allclose(result.information['values'].vector,
                            np.cos(model.values) ** 2, rtol=1e-12)
