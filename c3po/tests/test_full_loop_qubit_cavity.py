from c3po.control.envelopes import *
from c3po.cobj.component import ControlComponent as CtrlComp
from c3po.cobj.group import ComponentGroup as CompGroup
from c3po.control.control import Control as Control
from c3po.control.control import ControlSet as ControlSet

from c3po.control.generator import Device as Device
from c3po.control.generator import AWG as AWG
from c3po.control.generator import Mixer as Mixer
from c3po.control.generator import Generator as Generator

from c3po.utils.tf_utils import *

from c3po.cobj.component import *
from c3po.main.model import Model as mdl

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

import uuid
import copy
import pickle

import tensorflow as tf

import matplotlib.pyplot as plt


#########################
# USER FRONTEND SECTION #
#########################

redo_closed_loop = True
redo_open_loop = True

qubit_freq = 5e9*2*np.pi
qubit_inharm = -300e6*2*np.pi
qubit_lvls = 4

resonator_freq = 9e9*2*np.pi
resonator_lvls = 10

drive_amp = 80e-3 # 100 mV

mV_to_Amp = 2e9*np.pi

qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1

qubit_e = np.zeros([qubit_lvls, 1])
qubit_e[1] = 1


resonator_gs = np.zeros([resonator_lvls, 1])
resonator_gs[0] = 1

psi_init = tf.constant(
    np.kron(qubit_g, resonator_gs), dtype=tf.complex128
    )

psi_goal = tf.constant(
    np.kron(qubit_e, resonator_gs).T, dtype=tf.complex128
    )

##########################
#    END USER SECTION    #
##########################

env_group = CompGroup()
env_group.name = "env_group"
env_group.desc = "group containing all components of type envelop"


carr_group = CompGroup()
carr_group.name = "carr_group"
carr_group.desc = "group containing all components of type carrier"


carrier_parameters = {
    'freq' : 4.95e9 * 2 * np.pi
}

carr = CtrlComp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carr_group.get_uuid()]
)
carr_group.add_element(carr)


flattop_params1 = {
    'amp' : drive_amp,
    'T_up' : 3e-9,   # 3ns
    'T_down' : 9e-9, # 9ns
    'xy_angle' : 0.0,
    'freq_offset' : 0e6 * 2 * np.pi
}

params_bounds = {
    'amp' : [5e-3, 350e-3],
    'T_up' : [1e-9, 11e-9],
    'T_down' : [1e-9, 11e-9],
    'xy_angle' : [-np.pi, np.pi],
    'freq_offset' : [-0.2e9 * 2 * np.pi, 0.2e9 * 2 * np.pi]
}

def my_flattop(t, params):
    t_up = tf.cast(params['T_up'], tf.float64)
    t_down = tf.cast(params['T_down'], tf.float64)
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.math.erf((t - T1) / 1e-9)) / 2 * \
            (1 + tf.math.erf((-t + T2) / 1e-9)) / 2


p1 = CtrlComp(
    name = "pulse1",
    desc = "flattop comp 1 of signal 1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)

env_group.add_element(p1)

####
# Below code: For checking the single signal components
####

# t = np.linspace(0, 150e-9, int(150    #######################
    # Mtching model...   #
    #######################e-9*1e9))
# plt.plot(t, p1.get_shape_values(t))
# plt.plot(t, p2.get_shape_values(t))
# plt.show()


comps = []
comps.append(carr)
comps.append(p1)


ctrl = Control()
ctrl.name = "control1"
ctrl.t_start = 0.0
ctrl.t_end = 12e-9
ctrl.comps = comps


ctrls = ControlSet([ctrl])

opt_map = {
    'amp' : [(ctrl.get_uuid(), p1.get_uuid())],
#    'T_up' : [(ctrl.get_uuid(), p2.get_uuid())],
#    'T_down' : [(ctrl.get_uuid(), p2.get_uuid())],
#    'xy_angle' : [(ctrl.get_uuid(), p1.get_uuid())],
    'freq_offset' : [(ctrl.get_uuid(), p1.get_uuid())]
}

awg = AWG()
mixer = Mixer()


devices = {
    "awg" : awg,
    "mixer" : mixer
}

resolutions = {
    "awg" : 1e9,
    "sim" : 5e10
}


resources = [ctrl]


resource_groups = {
    "env" : env_group,
    "carr" : carr_group
}


gen = Generator()
gen.devices = devices
gen.resolutions = resolutions
gen.resources = resources
gen.resource_groups = resource_groups

output = gen.generate_signals()


set_tf_log_level(3)


q1 = Qubit(
    name = "Q1",
    desc = "Qubit 1",
    comment = "The one and only qubit in this chip",
    freq = qubit_freq,
    delta = qubit_inharm,
    hilbert_dim = qubit_lvls
    )

r1 = Resonator(
    name = "R1",
    desc = "Resonator 1",
    comment = "The resonator driving Qubit 1",
    freq = resonator_freq,
    hilbert_dim = resonator_lvls
    )

q1r1 = Coupling(
    name = "Q1-R1",
    desc = "Coupling between Resonator 1 and Qubit 1",
    comment = " ",
    connected = [q1.name, r1.name],
    strength = 150e6*2*np.pi
    )

drive = Drive(
    name = "D1",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1",
    connected = [q1.name]
    )

chip_elements = [
    q1,
    r1,
    q1r1,
    drive
    ]

initial_model = mdl(chip_elements, mV_to_Amp)
optimize_model = mdl(chip_elements, mV_to_Amp)

q2 = Qubit(
    name = "Q1",
    desc = "Qubit 2",
    comment = "The one and only qubit in this chip",
    freq = 5.05e9*2*np.pi,
    delta = qubit_inharm,
    hilbert_dim = qubit_lvls
    )

r2 = Resonator(
    name = "R1",
    desc = "Resonator 1",
    comment = "The resonator driving Qubit 1",
    freq = 0.95*resonator_freq,
    hilbert_dim = resonator_lvls
    )


q2r2 = Coupling(
    name = "Q1-R1",
    desc = "Coupling between Resonator 1 and Qubit 1",
    comment = " ",
    connected = [q2.name, r2.name],
    strength = 140e6*2*np.pi
    )

drive2 = Drive(
    name = "D2",
    desc = "Drive 2",
    comment = "Drive line 1 on qubit 1",
    connected = [q2.name]
    )

chip2_elements = [
    q2,
    r2,
    q2r2,
    drive2
    ]

real_model = mdl(chip2_elements, 0.72*2e9*np.pi)

rechenknecht = Opt()

opt_params = ctrls.get_corresponding_control_parameters(opt_map)
rechenknecht.opt_params = opt_params

sim = Sim(initial_model, gen, ctrls)

# Goal to drive on qubit 1
# U_goal = np.array(
#     [[0.+0.j, 1.+0.j, 0.+0.j],
#      [1.+0.j, 0.+0.j, 0.+0.j],
#      [0.+0.j, 0.+0.j, 1.+0.j]]
#     )

sim.model = optimize_model

exp_sim = Sim(real_model, gen, ctrls)

rechenknecht.simulate_noise = True

def evaluate_signals(pulse_params, opt_params):

    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal, psi_actual)

    return 1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)

def experiment_evaluate(pulse_params, opt_params):
    model_params = exp_sim.model.params
    U = exp_sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal, psi_actual)
    return 1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)

def match_model_psi(model_params, opt_params, pulse_params, result):

    U = sim.propagation(pulse_params, opt_params, model_params)

    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal, psi_actual)
    diff = (1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)) - result

    model_error = diff * diff

    return model_error


if redo_open_loop:
    print(
    """
    #######################
    # Optimizing pulse... #
    #######################
    """
    )

    def callback(xk):
        print(xk)

    settings = {} #'maxiter': 1}

    rechenknecht.optimize_controls(
        controls = ctrls,
        opt_map = opt_map,
        opt = 'lbfgs',
    #    opt = 'tf_grad_desc',
        settings = settings,
        calib_name = 'openloop',
        eval_func = evaluate_signals,
        callback = callback
        )


initial_spread = [5e-3, 0.1, 20e6*2*np.pi]

opt_settings = {
    'CMA_stds': initial_spread,
#    'maxiter' : 1,
#    'ftarget' : 1e-4,
    'popsize' : 20
}

if redo_closed_loop:
    print(
    """
    #######################
    # Calibrating...      #
    #######################
    """

    )

    rechenknecht.optimize_controls(
        controls = ctrls,
        opt_map = opt_map,
        opt = 'cmaes',
    #    opt = 'tf_grad_desc',
        settings = opt_settings,
        calib_name = 'closedloop',
        eval_func = experiment_evaluate
        )

# opt_sim = Sim(real_model, gen, ctrls)
# Fed: this here really scared me for a second

settings = {'maxiter': 100}

print(
"""
    #######################
    # Matching model...   #
    #######################
"""
)

if not redo_closed_loop:
    rechenknecht.load_history(previous_optim_log)

rechenknecht.learn_model(
    optimize_model,
    eval_func = match_model_psi,
    settings = settings,
    )

rechenknecht.save_history('improved_model.pickle')

# Enable this to rerun steps 1 and 2 with the improved model

# rechenknecht.optimize_controls(
#     controls = ctrls,
#     opt_map = opt_map,
#     opt = 'lbfgs',
# #    opt = 'tf_grad_desc',
#     settings = settings,
#     calib_name = 'openloop_2',
#     eval_func = evaluate_signals,
#     callback = callback
#     )
#
# rechenknecht.optimize_controls(
#     controls = ctrls,
#     opt_map = opt_map,
#     opt = 'cmaes',
# #    opt = 'tf_grad_desc',
#     settings = opt_settings,
#     calib_name = 'closedloop_2',
#     eval_func = experiment_evaluate
#     )
