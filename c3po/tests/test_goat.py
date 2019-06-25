"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po

from c3po.main.model import Model as mdl
from c3po.main.gate import Gate as gt

from c3po.utils.tf_utils import *
from c3po.evolution.propagation import *


import time




def plot_dynamics(u_list, ts, states):
    pop = []
    for si in states:
        for ti in range(len(ts)):
            pop.append(abs(u_list[ti][si][0] ** 2))
#        plt.plot(ts, pop)
    return pop


def plot_dynamics_sesolve(u_list, ts):
    pop = []
    for ti in range(len(ts)):
        pop.append(abs(u_list.states[ti].full().T[0] ** 2))
#    plt.plot(ts, pop)
    return pop



tf_log_level_info()

set_tf_log_level(2)

print("current log level: " + str(get_tf_log_level()))

sess = tf_setup()

print(" ")
print("Available tensorflow devices: ")
tf_list_avail_devices()



initial_parameters = {
        'qubit_1': {'freq': 6e9*2*pi},
        'cavity': {'freq': 9e9*2*pi}
        }

initial_couplings = {
        'q1_cav': {'strength': 150e6*2*pi}
        }

initial_hilbert_space = {
        'qubit_1': 2,
        'cavity': 5
        }

initial_model = mdl(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        "True"
        )

initial_model.set_tf_session(sess)



handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': 15e6*2*pi,
                            't_up': 5e-9,
                            't_down': 45e-9,
                            'xy_angle': 0,
                            'freq_offset': 0e6*2*pi
                            },
                        'func': my_flattop
                        }
                    }
                }
            }
        }


X_gate = gt('qubit_1', qt.sigmax())
def my_flattop(t, idx, guess):
    t_up = guess[idx['t_up']]
    t_down = guess[idx['t_down']]
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.erf((t - T1) / 1e-9)) / 2 * \
           (1 + tf.erf((-t + T2) / 1e-9)) / 2


pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse': {
                        'params': {
                            't_up': [2e-9, 98e-9],
                            't_down': [2e-9, 98e-9],
                            'freq_offset': [-1e9*2*pi, 1e9*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)

#######
# actually not needed just to be able to show the hamilton before propagation

# control_func = [q1_X_gate.get_control_fields('initial')]
# H = initial_model.get_Hamiltonian(control_func)
# print(H)

######


cflds, ts = X_gate.get_control_fields('initial')

hlist = initial_model.get_tf_Hamiltonian(cflds)

hlist[0]

n = int(1e4)
ts = np.linspace(0, 50e-9, n)

U0 = tensor(
    qeye(2),
    qeye(5)
)

tf_u = tf.constant(U0.full(), dtype=tf.complex128, name="u0")

# print(U0)

out2 = sesolve_pwc(hlist, U0, ts, sess, history=True)


print("time with tensorflow: " + str(half_time - start_time))

u_list = []
for i in range(0, len(out)):
    tmp = Qobj(out[i])
    u_list.append(tmp)

u_list2 = []
for i in range(0, len(out2)):
    tmp = Qobj(out2[i])
    u_list2.append(tmp)


pop1 = plot_dynamics(u_list, ts,[0])

# plt.plot(ts, pop1)
# plt.title("pwc_tf_1e4")
# plt.show()

pop2 = plot_dynamics(u_list2, ts, [0])

# plt.plot(ts, pop2)
# plt.title("pwc_tf_1e4")
# plt.show()



fig = plt.figure(1)
sp1 = plt.subplot(211)
name_str = "pwc_tf_%.2g" % n
sp1.title.set_text(name_str)
plt.plot(ts, pop1)

sp2 = plt.subplot(212)
name_str = "pwc_no_tf_%.2g" % n
sp2.title.set_text(name_str)
plt.plot(ts, pop2)

plt.show()
