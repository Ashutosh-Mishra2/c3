import numpy as np
from c3.c3objs import Quantity
from c3.libraries.envelopes import envelopes
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope, Carrier
from c3.utils.parsers import create_model, create_generator, create_gateset


model = create_model("test/test_model.cfg")
gen = create_generator("test/generator.cfg")
gates = create_gateset("test/instructions.cfg")


def test_subsystems() -> None:
    assert list(model.subsystems.keys()) == ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']


def test_couplings() -> None:
    assert list(model.couplings.keys()) == ['Q1-Q2', 'Q4-Q6', 'd1', 'd2']


def test_q6_freq() -> None:
    assert str(model.subsystems['Q6'].params['freq']) == '4.600 GHz 2pi '


def test_instructions() -> None:
    assert list(gates.instructions.keys()) == ["X90p", "Y90p", "X90m", "Y90m"]


def test_signal_generation() -> None:
    t_final = 7e-9   # Time for single qubit gates
    sideband = 50e6 * 2 * np.pi
    gauss_params_single = {
        'amp': Quantity(
            value=0.5,
            min_val=0.4,
            max_val=0.6,
            unit="V"
        ),
        't_final': Quantity(
            value=t_final,
            min_val=0.5 * t_final,
            max_val=1.5 * t_final,
            unit="s"
        ),
        'sigma': Quantity(
            value=t_final / 4,
            min_val=t_final / 8,
            max_val=t_final / 2,
            unit="s"
        ),
        'xy_angle': Quantity(
            value=0.0,
            min_val=-0.5 * np.pi,
            max_val=2.5 * np.pi,
            unit='rad'
        ),
        'freq_offset': Quantity(
            value=-sideband - 3e6 * 2 * np.pi,
            min_val=-56 * 1e6 * 2 * np.pi,
            max_val=-52 * 1e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'delta': Quantity(
            value=-1,
            min_val=-5,
            max_val=3,
            unit=""
        )
    }
    gauss_env_single = Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes['gaussian_nonorm']
    )

    carrier_parameters = {
        'freq': Quantity(
            value=5e9 * 2 * np.pi,
            min_val=4.5e9 * 2 * np.pi,
            max_val=6e9 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'framechange': Quantity(
            value=0.0,
            min_val=-np.pi,
            max_val=3 * np.pi,
            unit='rad'
        )
    }
    carr = Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters
    )

    X90p_q1 = Instruction(
        name="X90p",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )

    X90p_q1.add_component(gauss_env_single, "d1")
    X90p_q1.add_component(carr, "d1")

    gen.generate_signals(X90p_q1)
