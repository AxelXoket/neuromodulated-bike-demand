"""The four networks: where the state enters, the shared starting point, and the matched sizes."""
import unittest

import torch

from bike_demand import data, models

INPUT, STATE = len(data.INPUT_NAMES), len(data.STATE_NAMES)


def batch(rows: int = 16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, INPUT, generator=generator), torch.randn(rows, STATE, generator=generator)


def shaken(kind: str, width: int = 32) -> models.Net:
    """A network whose modulation weights are no longer zero, as after training."""
    torch.manual_seed(0)
    net = models.Net(kind, INPUT, STATE, width)
    for layer in net.modulation or []:
        for part in (layer.gamma, layer.beta):
            if part is not None:
                torch.nn.init.normal_(part.weight)
    return net


class WhereTheStateGoes(unittest.TestCase):
    def output_change(self, kind: str) -> float:
        net = shaken(kind)
        x, s = batch()
        with torch.no_grad():
            return float((net(x, s) - net(x, s + 1)).abs().max())

    def test_only_the_no_state_network_ignores_the_state(self):
        self.assertEqual(self.output_change("no-state"), 0.0)
        for kind in ("concat", "shift", "film"):
            with self.subTest(kind=kind):
                self.assertGreater(self.output_change(kind), 1e-3)

    def test_shift_and_film_start_as_the_same_network_without_state(self):
        for kind in ("shift", "film"):
            torch.manual_seed(1)
            modulated = models.Net(kind, INPUT, STATE, 32)
            plain = models.Net("no-state", INPUT, STATE, 32)
            plain.hidden.load_state_dict(modulated.hidden.state_dict())
            plain.out.load_state_dict(modulated.out.state_dict())
            x, s = batch()
            with self.subTest(kind=kind), torch.no_grad():
                torch.testing.assert_close(modulated(x, s), plain(x, s))
                torch.testing.assert_close(modulated(x, s), modulated(x, -3 * s))

    def test_gamma_scales_and_beta_shifts_each_hidden_unit(self):
        layer = models.Modulation(STATE, 4, scale=True)
        with torch.no_grad():
            layer.gamma.weight[:, 0] = torch.tensor([1.0, 0.0, -1.0, 2.0])
            layer.beta.weight[:, 1] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        h = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        s = torch.zeros(1, STATE)
        s[0, 0], s[0, 1] = 1.0, 2.0
        torch.testing.assert_close(layer(h, s), torch.tensor([[3.0, 3.0, 1.0, 13.0]]))

    def test_shift_only_adds_and_film_with_a_flat_gamma_is_shift(self):
        shift, film = models.Net("shift", INPUT, STATE, 16), models.Net("film", INPUT, STATE, 16)
        film.hidden.load_state_dict(shift.hidden.state_dict())
        film.out.load_state_dict(shift.out.state_dict())
        with torch.no_grad():
            for a, b in zip(shift.modulation, film.modulation):
                torch.nn.init.normal_(a.beta.weight)
                b.beta.weight.copy_(a.beta.weight)
        self.assertIsNone(shift.modulation[0].gamma)
        x, s = batch()
        with torch.no_grad():
            torch.testing.assert_close(film(x, s), shift(x, s))
            torch.nn.init.normal_(film.modulation[0].gamma.weight)
            self.assertGreater(float((film(x, s) - shift(x, s)).abs().max()), 1e-3)


    def test_the_second_layer_is_modulated_too(self):
        for kind in ("shift", "film"):
            net = models.Net(kind, INPUT, STATE, 16)
            with torch.no_grad():
                torch.nn.init.normal_(net.modulation[1].beta.weight)
            x, s = batch()
            with self.subTest(kind=kind), torch.no_grad():
                self.assertGreater(float((net(x, s) - net(x, s + 1)).abs().max()), 1e-3)


class Sizes(unittest.TestCase):
    def test_the_four_networks_have_the_same_number_of_parameters_within_one_percent(self):
        widths = models.widths(INPUT, STATE)
        counts = {k: models.parameter_count(models.Net(k, INPUT, STATE, w)) for k, w in widths.items()}
        self.assertEqual(widths["concat"], models.REFERENCE_WIDTH)
        self.assertEqual(set(counts), set(models.KINDS))
        for kind, count in counts.items():
            with self.subTest(kind=kind):
                self.assertLessEqual(abs(count - counts["concat"]) / counts["concat"], 0.01)
        self.assertEqual(len(set(widths.values())), len(models.KINDS))

    def test_the_output_is_one_number_per_row(self):
        x, s = batch(5)
        for kind in models.KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(models.Net(kind, INPUT, STATE, 8)(x, s).shape, (5,))

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            models.Net("state-as-mood", INPUT, STATE, 8)


if __name__ == "__main__":
    unittest.main()
