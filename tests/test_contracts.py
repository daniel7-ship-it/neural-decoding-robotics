import unittest
from dataclasses import replace

import numpy as np

from neuralduck.neural import VelocityCommand, effective_command, make_session_trials, fit_decoder
from neuralduck.microduck import MicroduckPolicy, create_simulation
from neuralduck.demo import RESEARCH, ROOT


class ContractTests(unittest.TestCase):
    def test_command_expiry_and_invalid_values(self):
        c = VelocityCommand(1, 0, 0, .65, .9, 1.3, "left")
        self.assertEqual(effective_command(c, 1.1)[2], .65)
        for when in (0, 1.3, 2, float("nan")):
            self.assertEqual(effective_command(c, when), (0, 0, 0))
        self.assertEqual(effective_command(replace(c, expires_at=float("nan")), 1.1), (0, 0, 0))

    def test_decode_separate_session_and_silencing(self):
        train = make_session_trials(0, 42, 8)
        test = make_session_trials(1, 42, 8)
        classifier = fit_decoder(train)
        predictions = classifier.predict(np.stack([t.features for t in test]))
        self.assertGreater(np.mean(predictions == [t.label for t in test]), .7)
        silenced = make_session_trials(1, 42, 8, silenced="left")
        for trial in silenced:
            np.testing.assert_array_equal(trial.activity[:48], 0)

    def test_policy_tracks_forward_and_opposite_turns(self):
        results = {}
        for name, command in (("forward", (.3, 0, 0)), ("left", (0, 0, 1.2)), ("right", (0, 0, -1.2))):
            model, data, bam = create_simulation(RESEARCH / "src/mjlab_microduck/robot/microduck/scene.xml")
            controller = MicroduckPolicy(model, data, bam, ROOT / "models/velstand.onnx")
            velocities, heights = [], []
            for _ in range(200):
                controller.step(command)
                velocities.append(controller.body_velocity())
                heights.append(float(data.qpos[2]))
                self.assertTrue(np.isfinite(data.qpos).all())
            results[name] = np.mean(velocities[50:], axis=0)
            self.assertGreater(min(heights[50:]), .07)
        self.assertGreater(results["forward"][0], .03)
        self.assertGreater(results["left"][2], .2)
        self.assertLess(results["right"][2], -.2)
        print("Measured forward/left/right velocities:", results)


if __name__ == "__main__":
    unittest.main()

