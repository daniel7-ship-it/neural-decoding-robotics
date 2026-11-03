import unittest

import numpy as np

from neuralduck.mapping import Settings, channels, combine, fit_models, record_session, score_models


class MappingTests(unittest.TestCase):
    def test_repeatable_paired_silencing_and_fixed_anatomy(self):
        original = record_session(42, 4, Settings(), 3)
        repeated = record_session(42, 4, Settings(), 3)
        silenced = record_session(42, 4, Settings(silence="left"), 3)
        np.testing.assert_array_equal(original.recordings, repeated.recordings)
        np.testing.assert_array_equal(original.labels, silenced.labels)
        np.testing.assert_array_equal(silenced.rates[:, :48], 0)
        self.assertTrue(set(channels(2)).issubset(channels(6)))
        fewer = record_session(42, 4, Settings(electrodes=6), 3)
        np.testing.assert_array_equal(fewer.recordings, original.recordings[:, :, channels(6)])

    def test_no_signal_cannot_reconstruct_individual_trials(self):
        train = combine([record_session(42, s, Settings(), 5) for s in (0, 1)])
        val = record_session(42, 2, Settings(), 5)
        test = record_session(42, 3, Settings(), 5)
        models = fit_models(train, val)
        normal = score_models(models, test)
        self.assertGreater(normal["reconstruction_skill_vs_constant"], .7)
        flat = test.features.copy()
        flat[:] = train.features.mean(axis=0)
        predicted, reconstructed, _, _ = models.infer(flat)
        np.testing.assert_allclose(reconstructed, np.repeat(reconstructed[:1], len(flat), axis=0), atol=1e-6)
        self.assertLess(len(np.unique(predicted)), 2)
        self.assertEqual(reconstructed.shape, test.rates.shape)

    def test_split_guard_and_parameters(self):
        train = record_session(42, 0, Settings(), 3)
        with self.assertRaises(ValueError):
            fit_models(train, train)
        for kwargs in ({"electrodes": 0}, {"electrodes": 25}, {"noise": -1}, {"shift": float("nan")}):
            with self.assertRaises(ValueError):
                Settings(**kwargs)


if __name__ == "__main__":
    unittest.main()
