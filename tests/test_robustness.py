import unittest

import numpy as np

from neuralduck.closed_loop import ContinuousCircuit, fit_decoders
from neuralduck.robustness import (CANDIDATES, CONDITIONS, CorrectedDecoder,
                                   RecordingDisturbance, locked_protocol, select_challenge, targets)


class RobustnessTests(unittest.TestCase):
    def test_protocol_split_and_bounded_search(self):
        spec = locked_protocol()
        sessions = spec['sessions']
        self.assertFalse(set(sessions['final_test']) & set(sessions['calibration'] +
                         [sessions['development'], sessions['reward_adaptation'],
                          sessions['regularization_validation']] + sessions['prior_final_do_not_reuse']))
        self.assertEqual(len(CANDIDATES), 6)
        self.assertGreaterEqual(spec['task']['rest_seconds'], 60)
        self.assertEqual(targets(42, 9), targets(42, 9))
        self.assertEqual(len(set(targets(42, 9))), 3)

    def test_paired_recording_changes_do_not_change_neurons(self):
        base = ContinuousCircuit(42, 6)
        shift = ContinuousCircuit(42, 6, disturbance=RecordingDisturbance(42, 6, 'shift_sudden'))
        lost = ContinuousCircuit(42, 6, disturbance=RecordingDisturbance(42, 6, 'loss_sudden'))
        noise = ContinuousCircuit(42, 6, disturbance=RecordingDisturbance(42, 6, 'noise_gradual'))
        for i in range(210):
            for stream in (base, shift, lost, noise):
                stream.sample(.8, i*.01)
        np.testing.assert_array_equal(base.rates, shift.rates)
        np.testing.assert_array_equal(base.rates, lost.rates)
        np.testing.assert_array_equal(base.rates, noise.rates)
        self.assertFalse(np.array_equal(base.observed[-1], shift.observed[-1]))
        np.testing.assert_array_equal(lost.observed[-1][12:], 0)
        self.assertTrue(np.isfinite(noise.observed[-1]).all())
        repeated = ContinuousCircuit(42, 6, disturbance=RecordingDisturbance(42, 6, 'noise_gradual'))
        for i in range(210):
            repeated.sample(.8, i*.01)
        np.testing.assert_array_equal(noise.observed[-1], repeated.observed[-1])
        gradual = RecordingDisturbance(42, 6, 'gain_gradual')
        self.assertEqual(gradual.magnitude(1), 0)
        self.assertAlmostEqual(gradual.magnitude(11), .5)
        self.assertEqual(gradual.magnitude(20), 1)
        before = RecordingDisturbance(42, 6, 'shift_sudden')
        baseline = base.electrodes.record(base.rates[None])[0]
        np.testing.assert_allclose(before.apply(base.rates, baseline, base.electrodes, 1.9), baseline,
                                   rtol=1e-6)

    def test_development_choice_uses_only_supplied_rows(self):
        rows = [{'condition': c, 'target_success_rate': 0 if c == 'loss_sudden' else 1}
                for c in CONDITIONS if c.endswith('sudden')]
        self.assertEqual(select_challenge(rows), 'loss_sudden')

    def test_inference_interface_and_adaptation_bound(self):
        model = fit_decoders(42)[0]['direct']
        wrapped = CorrectedDecoder(model, -1, 0)
        row = np.zeros(48)
        drive, _ = model.infer(row)
        changed, _ = wrapped.infer(row)
        self.assertAlmostEqual(changed, -drive)
        with self.assertRaises(TypeError):
            wrapped.infer(row, truth=np.ones(96))
        with self.assertRaises(ValueError):
            wrapped.infer(np.full(48, np.nan))
        self.assertTrue(all(abs(g)<=1.4 and abs(b)<=.15 for g, b in CANDIDATES))


if __name__ == '__main__':
    unittest.main()
