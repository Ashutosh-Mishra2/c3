import unittest
import numpy as np
import c3po.component as component
import c3po.control as control
import c3po.envelopes as envelopes
import c3po.generator as generator
from c3po.model import Model as mdl


class TestModel(unittest.TestCase):
    def test_qubit(self):
        q1 = component.Qubit(
            name="Q1",
            desc="Qubit 1",
            comment="",
            freq=5e9 * 2 * np.pi,
            anhar=-150e6 * 2 * np.pi,
            hilbert_dim=3
        )
        chip_elements = [q1]
        model = mdl(chip_elements, 0.7e9 * 2 * np.pi)
        self.assertCountEqual(
            model.params, [5e9 * 2 * np.pi, -150e6 * 2 * np.pi]
        )

        self.assertCountEqual(
            model.params_desc, [['Q1', 'freq'], ['Q1', 'anhar']]
        )


class TestControl(unittest.TestCase):
    def test_parameter_set_get(self):
        gauss_params = {
            'amp': 150e6 * 2 * np.pi,
            't_final': 8e-9,
            'xy_angle': 0.0,
            'freq_offset': 0e6 * 2 * np.pi
        }
        gauss_bounds = {
            'amp': [0.01 * 150e6 * 2 * np.pi, 1.5 * 150e6 * 2 * np.pi],
            't_final': [7e-9, 12e-9],
            'xy_angle': [-1 * np.pi / 2, 1 * np.pi / 2],
            'freq_offset': [-100 * 1e6 * 2 * np.pi, 100 * 1e6 * 2 * np.pi]
        }
        gauss_env = component.Envelope(
            name="gauss",
            desc="Gaussian comp 1 of signal 1",
            params=gauss_params,
            bounds=gauss_bounds,
            shape=envelopes.gaussian
        )
        carrier_parameters = {
            'freq': 5.95e9 * 2 * np.pi
        }
        carrier_bounds = {
            'freq': [5e9 * 2 * np.pi, 7e9 * 2 * np.pi]
        }
        carr = component.Carrier(
            name="carrier",
            desc="Frequency of the local oscillator",
            params=carrier_parameters,
            bounds=carrier_bounds
        )
        ctrl = control.Control(
            name="line1",
            t_start=0.0,
            t_end=10e-9,
            comps=[carr, gauss_env]
        )
        ctrls = control.ControlSet([ctrl])
        opt_map = [
            ('line1', 'gauss', 'amp'),
            ('line1', 'gauss', 'freq_offset')
        ]
        opt_params = ctrls.get_corresponding_control_parameters(opt_map)
        values, bounds = ctrls.get_values_bounds(opt_params)
        self.assertCountEqual(
            bounds, [
                [0.01 * 150e6 * 2 * np.pi, 1.5 * 150e6 * 2 * np.pi],
                [-100 * 1e6 * 2 * np.pi, 100 * 1e6 * 2 * np.pi]
            ]
        )
        self.assertCountEqual(
            np.array(values),
            np.array([150e6 * 2 * np.pi, 0.0])
        )

    def test_devices(self):
        lo = generator.LO(resolution=5e10)
        awg = generator.AWG(resolution=1.2e9)
        mixer = generator.Mixer()
        devices = {
            "lo": lo,
            "awg": awg,
            "mixer": mixer
        }
        gen = generator.Generator(devices)
        self.assertDictEqual(gen.devices, devices)


if __name__ == '__main__':
    unittest.main()
