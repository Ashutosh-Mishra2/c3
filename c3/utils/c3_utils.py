"""General utilities for C3 functions."""

import numpy as np
import copy

import c3.generator.devices as devices
import c3.signal.pulse as pulse
import c3.signal.gates as gates
import c3.libraries.envelopes as envelopes

from c3.c3objs import Quantity as Qty


def convert_to_pwc_batch(instr, num_batches, awg_res=2e9):
    # ##specify the awg resolution if otherwise
    ts = (
        devices.AWG(name="awg", resolution=awg_res, outputs=1)
        .create_ts(instr.t_start, instr.t_end, centered=False)
        .numpy()
    )
    ts_centered = (
        devices.AWG(name="awg", resolution=awg_res, outputs=1)
        .create_ts(instr.t_start, instr.t_end, centered=True)
        .numpy()
    )

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
                        chan, ts_centered[n * batch_size : (n + 1) * batch_size], index
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
                        min_val=-2 * np.abs(amp),
                        max_val=2 * np.abs(amp),
                        unit="",
                    ),
                    "quadrature": Qty(
                        value=signal["quadrature"],
                        min_val=-2 * np.abs(amp),
                        max_val=2 * np.abs(amp),
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
