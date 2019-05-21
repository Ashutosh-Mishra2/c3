"""C3PO configuration file"""

import sys
import zmq
from numpy import pi
import numpy as np
import qutip as qt
import uuid
import c3po

"""
Device specific setup goes here. We need to provide a function that takes a
gate, evaluates it in the physical machine and returns a figure of merit.
"""


def evaluate_pulse(gate, q=None):
    global search_id
    pulse_id = str(uuid.uuid4())
    if q is None:
        print('No parametrization specified. Using initial guess.')
        q = gate.parameters['initial']
    gate.print_pulse(q)
    gate.plot_control_fields(q)
    pulse = gate.get_IQ(q)
    ts = np.linspace(0, 50e-9, 51)
    pulse['I'] = list(map(pulse['I'], ts))
    pulse['Q'] = list(map(pulse['Q'], ts))
    request = {
            'search_id': search_id,
            'pulse_id': pulse_id,
            'pulse': pulse,
            'fidelity': 0,
            'do_stop': False
            }
    try:
        socketreq.send_json(request)
        receive_message = socketreq.recv_string()
    except:
        print(f"failed", flush=True)
        socketreq.unbind(calibration_daemon_experiment_URI)
        socketrep.close()
        sys.exit()

    if not request['do_stop']: # MAKES NO SENSE
        try:
            reply = socketrep.recv_json()
            socketrep.send_string(search_id)
        except:
            print('no connection')
            socketreq.unbind(calibration_daemon_experiment_URI)
            socketrep.close()
            sys.exit()

    # Make sure it's the right one
    if (reply['pulse_id'] != request['pulse_id']) or (reply['search_id'] != request['search_id']):  # MAKES NO SENSE
        try:
            print(f"request: {request}", file=sys.stderr)
            print(f"reply:   {reply}", file=sys.stderr)
        except:
            print("Wrong pulse_id or search_id")
            socketreq.unbind(calibration_daemon_experiment_URI)
            socketrep.close()
            sys.exit()

    # Extract the fidelity
    current_fidelity = reply['fidelity']
    return 1-current_fidelity


"""===Communication setup==="""
# Replace 'localhost' with the name or IP of the LabView machine
calibration_daemon_experiment_URI = "tcp://127.0.0.1:5559"
calibration_daemon_searcher_URI = "tcp://127.0.0.1:5560"
rcvtimeout = 30000
search_id = str(uuid.uuid4())
print(f"I am the calibration searcher.\nMy ID for this run is {search_id}\n")

# Start communication
print(f"Connecting to client {calibration_daemon_experiment_URI} ... ", flush=True, end='')
context = zmq.Context()
socketrep = context.socket(zmq.REP)
try:
    socketrep.bind(calibration_daemon_searcher_URI)
except zmq.ZMQError:
    print("Socket already in use. Rebinding ...")
    socketrep.unbind(calibration_daemon_searcher_URI)
    socketrep.bind(calibration_daemon_searcher_URI)
    print("Done.")

socketreq = context.socket(zmq.REQ)
socketreq.setsockopt(zmq.LINGER, 0)  # NEW Added by Edwar to flush the queue
socketreq.connect(calibration_daemon_experiment_URI)

socketrep.RCVTIMEO = rcvtimeout  # added timeout to kill rcv if nothing comes
socketreq.RCVTIMEO = rcvtimeout
print(f"done\n\n", flush=True)

q1_X_gate = c3po.Gate('qubit_1', qt.sigmax())

handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                # 'target': 'q1', # add here?
                'pulses': {
                    'pulse1': {
                        'amp': 15e6*2*pi,
                        't_up': 5e-9,
                        't_down': 45e-9,
                        'xy_angle': 0
                        }
                    }
                }
            }
        }

pulse_bounds = {
        'control1': {
            'carrier1': {
                'freq': [1e9*2*pi, 15e9*2*pi],
                'pulses': {
                    'pulse1': {
                        'amp':  [1e3*2*pi, 10e9*2*pi],
                        't_up': [2e-9, 98e-9],
                        't_down': [2e-9, 98e-9],
                        'xy_angle': [-pi, pi]
                        }
                    }
                }
            }
        }

q1_X_gate.set_parameters('initial', handmade_pulse)
q1_X_gate.set_bounds(pulse_bounds)

fridge = c3po.fidelity.measurement.Experiment(evaluate_pulse)
fridge.calibrate(q1_X_gate)
