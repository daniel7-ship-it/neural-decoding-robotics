import unittest

import numpy as np

from neuralduck.closed_loop import (CommandGate, ContinuousCircuit, DT, Episode,
                                    fit_decoders, sensory_input)
from neuralduck.neural import effective_command


class ContinuousTests(unittest.TestCase):
    def test_persistent_state_and_causal_window(self):
        stream = ContinuousCircuit(42, 4)
        for i in range(19):
            stream.sample(1., i*.01)
            self.assertIsNone(stream.features())
        stream.sample(1., .19)
        first = stream.features().copy()
        previous = stream.rates.copy()
        stream.sample(0., .20)
        self.assertGreater(stream.rates[:48].mean(), .2)
        self.assertLess(np.linalg.norm(stream.rates-previous), .2*np.linalg.norm(previous))
        self.assertEqual(len(stream.observed), 20)
        self.assertAlmostEqual(stream.times[0], .01)
        self.assertFalse(np.array_equal(first, stream.features()))

    def test_hysteresis_stop_and_expiry(self):
        gate = CommandGate()
        self.assertEqual(gate.update(1., 0.).yaw_rate, 0)
        go = gate.update(1., .1)
        self.assertGreater(go.yaw_rate, 0)
        self.assertEqual(effective_command(go, .35), (0, 0, 0))
        self.assertEqual(gate.update(-1., .2).yaw_rate, 0)
        self.assertLess(gate.update(-1., .3).yaw_rate, 0)
        self.assertEqual(gate.update(0., .4).yaw_rate, 0)
        self.assertEqual(sensory_input(1., False), 0)

    def test_real_policy_feedback_and_inference_boundary(self):
        models, _ = fit_decoders(42)
        for model in models.values():
            with self.assertRaises(ValueError):
                model.infer(np.full(48, np.nan))
        episode = Episode(models['direct'], 42, 4)
        self.assertIsNone(episode.metrics()['target_success_rate'])
        self.assertIsNone(episode.metrics()['observation_to_command_seconds_mean'])
        for _ in range(200):
            episode.step()
        self.assertTrue(np.isfinite(episode.data.qpos).all())
        self.assertTrue(any(r['yaw_command'] > 0 for r in episode.trace))
        self.assertTrue(all(d['latest_sample'] < d['time'] for d in episode.decisions))
        self.assertTrue(all(abs(d['observation_to_command_sim_seconds']-.2) < 1e-9
                            for d in episode.decisions))
        self.assertLess(episode.trace[-1]['error'], 1.)


if __name__ == '__main__':
    unittest.main()
