"""Miscellaneous, general utilities."""
import time
import os
import tempfile
import numpy as np
from typing import List, Tuple
import warnings
import copy

import c3.generator.devices as devices
import c3.signal.gates as gates
from c3.c3objs import Quantity as Qty
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes


# SYSTEM AND SETUP
def log_setup(data_path: str = None, run_name: str = "run") -> str:
    """
    Make sure the file path to save data exists. Create an appropriately named
    folder with date and time. Also creates a symlink "recent" to the folder.

    Parameters
    ----------
    data_path : str
        File path of where to store any data.
    run_name : str
        User specified name for the run.

    Returns
    -------
    str
        The file path to store new data.

    """
    if data_path:
        data_path = os.path.abspath(data_path)
    else:
        data_path = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    pwd = os.path.join(
        data_path, run_name, time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime())
    )
    while os.path.exists(pwd):
        time.sleep(1)
        pwd = os.path.join(
            data_path, run_name, time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime())
        )

    os.makedirs(pwd)
    recent = os.path.join(data_path, "recent")
    replace_symlink(pwd, recent)
    return os.path.join(pwd, "")


def replace_symlink(path: str, alias: str) -> None:
    """Create a symbolic link."""
    try:
        os.remove(alias)
    except FileNotFoundError:
        pass
    except PermissionError:
        warnings.warn("Could not remove symlink")
    try:
        os.symlink(path, alias)
    except FileExistsError:
        pass
    except OSError:
        warnings.warn("OSError encountered while creating symlink")


# NICE PRINTING FUNCTIONS
def eng_num(val: float) -> Tuple[float, str]:
    """Convert a number to engineering notation by returning number and prefix."""
    if np.array(val).size > 1:
        return np.array(val), ""
    if np.isnan(val):
        return np.nan, "NaN"
    big_units = ["", "K", "M", "G", "T", "P", "E", "Z"]
    small_units = ["m", "µ", "n", "p", "f", "a", "z"]
    sign = 1
    if val == 0:
        return 0, ""
    if val < 0:
        val = -val
        sign = -1
    tmp = np.log10(val)
    idx = int(tmp // 3)
    if tmp < 0:
        if np.abs(idx) > len(small_units):
            return val * (10 ** (3 * len(small_units))), small_units[-1]
        prefix = small_units[-(idx + 1)]
    else:
        if np.abs(idx) > len(big_units) - 1:
            return val * (10 ** (-3 * (len(big_units) - 1))), big_units[-1]
        prefix = big_units[idx]

    return sign * (10 ** (tmp % 3)), prefix


def num3str(val: float, use_prefix: bool = True) -> str:
    """Convert a number to a human readable string in engineering notation."""
    if np.array(val).size > 1:
        return np.array2string(val, precision=3)
    if use_prefix:
        num, prefix = eng_num(val)
        formatted_string = f"{num:.3f} " + prefix
    else:
        formatted_string = f"{val:.3} "
    return formatted_string


# USER INTERACTION
def ask_yn() -> bool:
    """Ask for y/n user decision in the command line."""
    asking = True
    text = input("(y/n): ")
    if text == "y":
        asking = False
        boolean = True
    elif text == "n":
        asking = False
        boolean = False
    while asking:
        text = input("Please write y or n and press enter: ")
        if text == "y":
            asking = False
            boolean = True
        elif text == "n":
            asking = False
            boolean = False
    return boolean


def deprecated(message: str):
    """Decorator for deprecating functions

    Parameters
    ----------
    message : str
        Message to display along with DeprecationWarning

    Examples
    --------
    Add a :code:`@deprecated("message")` decorator to the function::

        @deprecated("Using standard width. Better use gaussian_sigma.")
        def gaussian(t, params):
            ...

    """

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function. {}".format(func.__name__, message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


def flatten(lis: List, ltypes=(list, tuple)) -> List:
    """Flatten lists of arbitrary lengths
    https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html

    Parameters
    ----------
    lis : List
        The iterable to flatten
    ltypes : tuple, optional
        Possibly the datatype of the iterable, by default (list, tuple)

    Returns
    -------
    List
        Flattened list
    """
    ltype = type(lis)
    lis = list(lis)
    i = 0
    while i < len(lis):
        while isinstance(lis[i], ltypes):
            if not lis[i]:
                lis.pop(i)
                i -= 1
                break
            else:
                lis[i : i + 1] = lis[i]
        i += 1
    return ltype(lis)


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
