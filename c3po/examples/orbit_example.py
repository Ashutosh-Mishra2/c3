"""Script to run orbit optimization."""

import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from single_qubit import create_chip_model, create_generator, create_gates

# System
qubit_freq = 6e9 * 2 * np.pi
qubit_anhar = -100e6 * 2 * np.pi
qubit_lvls = 6
drive_ham = hamiltonians.x_drive
v_hz_conversion = 2e9 * np.pi
t_final = 10e-9

# Simulation variables
sim_res = 1e11
awg_res = 1e9  # 1.2GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gates = create_gates(t_final)

# Simulation class and fidelity function
sim = Sim(model, gen, gates)
opt_map = gates.list_parameters()


# Simulation class and fidelity function
sim = Sim(model, gen, gates)

qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
qubit_e = np.zeros([qubit_lvls, 1])
qubit_e[1] = 1
ket_init = tf.constant(qubit_g, tf.complex128)
bra_goal = tf.constant(qubit_e.T, tf.complex128)

# Optimizer object
opt = Opt()
opt_map = [
    [('X90p', 'line1', 'gauss', 'amp'),
     ('Y90p', 'line1', 'gauss', 'amp'),
     ('X90m', 'line1', 'gauss', 'amp'),
     ('Y90m', 'line1', 'gauss', 'amp')],
    [('X90p', 'line1', 'gauss', 'freq_offset'),
     ('Y90p', 'line1', 'gauss', 'freq_offset'),
     ('X90m', 'line1', 'gauss', 'freq_offset'),
     ('Y90m', 'line1', 'gauss', 'freq_offset')],
    [('X90p', 'line1', 'gauss', 'xy_angle')],
    [('Y90p', 'line1', 'gauss', 'xy_angle')],
    [('X90m', 'line1', 'gauss', 'xy_angle')],
    [('Y90m', 'line1', 'gauss', 'xy_angle')],
]

def evaluate_signals(pulse_values: list, opt_map: list):
    model_params = sim.model.params
    U = sim.propagation(pulse_values, opt_map, model_params)
    psi_actual = tf.matmul(U, ket_init)
    overlap = tf.matmul(bra_goal, psi_actual)
    return 1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)

opt.optimize_controls(
    controls=gates,
    opt_map=opt_map,
    opt='lbfgs',
    calib_name='openloop',
    eval_func=evaluate_signals
    )
