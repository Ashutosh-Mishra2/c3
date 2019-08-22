"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po

from c3po.main.model import Model as mdl
from c3po.main.gate import Gate as gt
from c3po.main.measurement import Simulation as sim

from c3po.utils.tf_utils import *
from c3po.evolution.propagation import *

import time

from tensorflow.python import debug as tf_debug

tf_log_level_info()

set_tf_log_level(2)

print("current log level: " + str(get_tf_log_level()))

sess = tf_setup()

print(" ")
print("Available tensorflow devices: ")
tf_list_avail_devices()

initial_parameters = {
        'q1': {'freq': 6e9*2*np.pi, 'delta': 100e6 * 2 * np.pi},
        'r1': {'freq': 9e9*2*np.pi}
        }
initial_couplings = {
        ('q1', 'r1'): {'strength': 150e6*2*np.pi}
        }
initial_hilbert_space = {
        'q1': 2,
        'r1': 5
        }
comp_hilbert_space = {
        'q1': 2,
        'r1': 5
        }
model_types = {
        'components': {
            'q1': c3po.utils.hamiltonians.duffing,
            'r1': c3po.utils.hamiltonians.resonator},
        'couplings': {
            ('q1', 'r1'): c3po.utils.hamiltonians.int_XX},
        'drives': {
            'q1': c3po.utils.hamiltonians.drive},
        }

initial_model = mdl(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        model_types
        )


U_goal = tensor(
    basis(2,1),
    basis(5,0)
).full()

U0 = tensor(
    basis(2,0),
    basis(5,0)
).full()

X_gate = gt('qubit_1', U_goal, T_final=30e-9)
pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': [15e6*2*pi, 65e6*2*pi],
                            't_up': [2e-9, 48e-9],
                            't_down': [2e-9, 48e-9],
                            'xy_angle': [-pi, pi],
                            'freq_offset': [-20e6*2*pi, 20e6*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)

rechenknecht = sim(initial_model, sesolve_pwc, sess)

res = 50e9
rechenknecht.resolution=res
