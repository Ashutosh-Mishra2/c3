"""General utilities for C3 functions."""

import numpy as np
import copy
import tensorflow as tf

import c3.signal.pulse as pulse
import c3.signal.gates as gates
import c3.libraries.envelopes as envelopes

from c3.c3objs import Quantity as Qty


def calc_slice_num(
    awg_res: float = 2e9, t_start: float = 0.0, t_end: float = 0.0
) -> int:
    """
    Effective number of time slices given start, end and resolution.
    Same as c3.generator.devices.AWG.calc_slice_num but return numpy array.

    Parameters
    ----------
    awg_res: float
        Resolution of AWG.
    t_start: float
        Starting time for this device.
    t_end: float
        End time for this device.
    """
    slice_num = int(np.round(np.abs(t_start - t_end) * awg_res))
    return slice_num


def create_ts(
    awg_res: float = 2e9, t_start: float = 0, t_end: float = 0, centered: bool = True
) -> tf.constant:
    """
    Compute time samples.
    Same as c3.generator.devices.AWG.create_ts but return numpy array.

    Parameters
    ----------
    awg_res: float
        Resolution of AWG.
    t_start: float
        Starting time for this device.
    t_end: float
        End time for this device.
    centered: boolean
        Sample in the middle of an interval, otherwise at the beginning.
    """

    # Slice num can change between pulses
    slice_num = calc_slice_num(awg_res, t_start, t_end)
    dt = 1 / awg_res
    # TODO This type of centering does not guarantee zeros at the ends
    if centered:
        offset = dt / 2
        num = slice_num
    else:
        offset = 0
        num = slice_num + 1
    t_start = t_start + offset
    t_end = t_end - offset
    ts = np.linspace(t_start, t_end, num)
    return ts


def convert_to_pwc_batch(instr, num_batches, awg_res=2e9):
    # ##specify the awg resolution if otherwise
    ts = create_ts(awg_res, instr.t_start, instr.t_end, centered=False)
    ts_centered = create_ts(awg_res, instr.t_start, instr.t_end, centered=True)

    channels = [i for i in instr.comps]
    pwc_gates_list = []
    batch_size = int(ts.shape[0] / num_batches)

    for n in range(num_batches):
        for chan in instr.comps:
            indices = [
                instr.comps[chan][key].index
                for key in instr.comps[chan]
                if isinstance(instr.comps[chan][key], pulse.Envelope)
            ]
            pulse_names = [
                i
                for i in instr.comps[chan].keys()
                if isinstance(instr.comps[chan][i], pulse.Envelope)
            ]

            if n == num_batches - 1:
                pwc_gate = gates.Instruction(
                    name=f"{instr.name}_pwc_{n}",
                    targets=instr.targets,
                    t_start=ts[n * batch_size],
                    t_end=ts[-1],
                    channels=channels,
                )
            else:
                pwc_gate = gates.Instruction(
                    name=f"{instr.name}_pwc_{n}",
                    targets=instr.targets,
                    t_start=ts[n * batch_size],
                    t_end=ts[(n + 1) * batch_size],
                    channels=channels,
                )

            for index in indices:
                pulse_name = pulse_names[index - 1]
                if n == num_batches - 1:
                    signal, norm = instr.get_awg_signal(
                        chan, ts_centered[n * batch_size :], index
                    )
                    t_final = ts[-1]
                else:
                    signal, norm = instr.get_awg_signal(
                        chan, ts_centered[n * batch_size : (n + 1) * batch_size], index
                    )
                    t_final = ts[(n + 1) * batch_size]

                non_pwc_pulse = instr.comps[chan][pulse_name]
                amp = non_pwc_pulse.params["amp"].get_value()

                pulse_params = {
                    "inphase": Qty(
                        value=signal["inphase"],
                        min_val=-2 * tf.abs(amp),
                        max_val=2 * tf.abs(amp),
                        unit="",
                    ),
                    "quadrature": Qty(
                        value=signal["quadrature"],
                        min_val=-2 * tf.abs(amp),
                        max_val=2 * tf.abs(amp),
                        unit="",
                    ),
                    "t_final": Qty(value=t_final),
                }

                pwc_pulse = pulse.Envelope(
                    name=non_pwc_pulse.name,
                    desc=non_pwc_pulse.desc,
                    params=pulse_params,
                    shape=envelopes.pwc,
                    index=index,
                )

                pwc_gate.add_component(copy.deepcopy(pwc_pulse), chan)

                carrier = instr.comps[chan][f"carrier{index}"]
                pwc_gate.add_component(copy.deepcopy(carrier), chan)

        pwc_gates_list.append(pwc_gate)

    return pwc_gates_list
