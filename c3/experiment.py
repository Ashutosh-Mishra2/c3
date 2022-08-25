"""
Experiment class that models and simulates the whole experiment.

It combines the information about the model of the quantum device, the control stack
and the operations that can be done on the device.

Given this information an experiment run is simulated, returning either processes,
states or populations.
"""

import os
import copy
import pickle
import itertools
import hjson
import numpy as np
import tensorflow as tf
from typing import Dict
import time

from c3.c3objs import hjson_encode, hjson_decode
from c3.generator.generator import Generator
from c3.parametermap import ParameterMap
from c3.signal.gates import Instruction
from c3.model import Model
from c3.utils.tf_utils import (
    tf_state_to_dm,
    tf_super,
    tf_vec_to_dm,
    _tf_matmul_n_even,
    _tf_matmul_n_odd,
    calculate_expectation_value
)

from c3.libraries.propagation import unitary_provider, state_provider

from c3.utils.qt_utils import perfect_single_q_parametric_gate
from scipy.integrate import ode


class Experiment:
    """
    It models all of the behaviour of the physical experiment, serving as a
    host for the individual parts making up the experiment.

    Parameters
    ----------
    pmap: ParameterMap
        including
        model: Model
            The underlying physical device.
        generator: Generator
            The infrastructure for generating and sending control signals to the
            device.
        gateset: GateSet
            A gate level description of the operations implemented by control
            pulses.

    """

    def __init__(self, pmap: ParameterMap = None, prop_method=None, sim_res=100e9):
        self.pmap = pmap
        self.opt_gates = None
        self.propagators: Dict[str, tf.Tensor] = {}
        self.partial_propagators: Dict = {}
        self.created_by = None
        self.logdir: str = ""
        self.propagate_batch_size = None
        self.use_control_fields = True
        self.overwrite_propagators = True  # Keep only currently computed propagators
        self.compute_propagators_timestamp = 0
        self.stop_partial_propagator_gradient = True
        self.evaluate = self.evaluate_legacy
        self.sim_res = sim_res
        self.prop_method = prop_method
        self.set_prop_method(prop_method)

    def set_prop_method(self, prop_method=None) -> None:
        """
        Configure the selected propagation method by either linking the function handle or
        looking it up in the library.
        """
        if prop_method is None:
            self.propagation = unitary_provider["pwc"]
            if self.pmap is not None:
                self._compute_folding_stack()
        elif isinstance(prop_method, str):
            try:
                self.propagation = unitary_provider[prop_method]
            except KeyError:
                self.propagation = state_provider[prop_method]
        elif callable(prop_method):
            self.propagation = prop_method

    def _compute_folding_stack(self):
        self.folding_stack = {}
        for instr in self.pmap.instructions.values():
            n_steps = int((instr.t_end - instr.t_start) * self.sim_res)
            if n_steps not in self.folding_stack:
                stack = []
                while n_steps > 1:
                    if not n_steps % 2:  # is divisable by 2
                        stack.append(_tf_matmul_n_even)
                    else:
                        stack.append(_tf_matmul_n_odd)
                    n_steps = np.ceil(n_steps / 2)
                self.folding_stack[
                    int((instr.t_end - instr.t_start) * self.sim_res)
                ] = stack

    def enable_qasm(self) -> None:
        """
        Switch the sequencing format to QASM. Will become the default.
        """
        self.evaluate = self.evaluate_qasm

    def set_created_by(self, config):
        """
        Store the config file location used to created this experiment.
        """

        self.created_by = config

    def load_quick_setup(self, filepath: str) -> None:
        """
        Load a quick setup file.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read(), object_pairs_hook=hjson_decode)
        self.quick_setup(cfg, os.path.dirname(filepath))

    def quick_setup(self, cfg, base_dir: str = None) -> None:
        """
        Load a quick setup cfg and create all necessary components.

        Parameters
        ----------
        cfg : Dict
            Configuration options

        """

        def make_absolute(filename: str) -> str:
            if base_dir or os.path.isabs(filename):
                return os.path.join(base_dir, filename)
            else:
                return filename

        model = Model()
        model.read_config(make_absolute(cfg["model"]))
        gen = Generator()
        gen.read_config(make_absolute(cfg["generator"]))

        single_gate_time = cfg["single_qubit_gate_time"]
        v2hz = cfg["v2hz"]
        instructions = []
        sideband = cfg.pop("sideband", None)
        for gate_name, props in cfg["single_qubit_gates"].items():
            target_qubit = model.subsystems[props["qubits"]]
            instr = Instruction(
                name=props["name"],
                targets=[model.names.index(props["qubits"])],
                t_start=0.0,
                t_end=single_gate_time,
                channels=[target_qubit.drive_line],
            )
            instr.quick_setup(
                target_qubit.drive_line,
                target_qubit.params["freq"].get_value() / 2 / np.pi,
                single_gate_time,
                v2hz,
                sideband,
            )
            instructions.append(instr)

        for gate_name, props in cfg["two_qubit_gates"].items():
            qubit_1 = model.subsystems[props["qubit_1"]]
            qubit_2 = model.subsystems[props["qubit_2"]]
            instr = Instruction(
                name=props["name"],
                targets=[
                    model.names.index(props["qubit_1"]),
                    model.names.index(props["qubit_2"]),
                ],
                t_start=0.0,
                t_end=props["gate_time"],
                channels=[qubit_1.drive_line, qubit_2.drive_line],
            )
            instr.quick_setup(
                qubit_1.drive_line,
                qubit_1.params["freq"].get_value() / 2 / np.pi,
                props["gate_time"],
                v2hz,
                sideband,
            )
            instr.quick_setup(
                qubit_2.drive_line,
                qubit_2.params["freq"].get_value() / 2 / np.pi,
                props["gate_time"],
                v2hz,
                sideband,
            )
            instructions.append(instr)

        self.sim_res = 100e9
        self.pmap = ParameterMap(instructions, generator=gen, model=model)
        self.set_prop_method()

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a Model object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read(), object_pairs_hook=hjson_decode)
        self.from_dict(cfg)

    def from_dict(self, cfg: Dict) -> None:
        """
        Load experiment from dictionary
        """
        model = Model()
        model.fromdict(cfg["model"])
        generator = Generator()
        generator.fromdict(cfg["generator"])
        pmap = ParameterMap(model=model, generator=generator)
        pmap.fromdict(cfg["instructions"])
        if "options" in cfg:
            for k, v in cfg["options"].items():
                self.__dict__[k] = v
        self.pmap = pmap
        self.sim_res = cfg.pop("sim_res", 100e9)
        self.set_prop_method()

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file, default=hjson_encode)

    def asdict(self) -> Dict:
        """
        Return a dictionary compatible with config files.
        """
        exp_dict: Dict[str, Dict] = {}
        exp_dict["instructions"] = {}
        for name, instr in self.pmap.instructions.items():
            exp_dict["instructions"][name] = instr.asdict()
        exp_dict["model"] = self.pmap.model.asdict()
        exp_dict["generator"] = self.pmap.generator.asdict()
        exp_dict["options"] = {
            "propagate_batch_size": self.propagate_batch_size,
            "use_control_fields": self.use_control_fields,
            "overwrite_propagators": self.overwrite_propagators,
            "stop_partial_propagator_gradient": self.stop_partial_propagator_gradient,
        }
        exp_dict["sim_res"] = self.sim_res
        return exp_dict

    def __str__(self) -> str:
        return hjson.dumps(self.asdict(), default=hjson_encode)

    def evaluate_legacy(self, sequences, psi_init: tf.Tensor = None):
        """
        Compute the population values for a given sequence of operations.

        Parameters
        ----------
        sequences: str list
            A list of control pulses/gates to perform on the device.

        psi_init: tf.Tensor
            A tensor containing the initial statevector

        Returns
        -------
        list
            A list of populations

        """
        model = self.pmap.model
        if psi_init is None:
            psi_init = model.get_init_state()
        populations = []
        for sequence in sequences:
            psi_t = copy.deepcopy(psi_init)
            for gate in sequence:
                psi_t = tf.matmul(self.propagators[gate], psi_t)

            pops = self.populations(psi_t, model.lindbladian)
            populations.append(pops)
        return populations

    def evaluate_qasm(self, sequences, psi_init: tf.Tensor = None):
        """
        Compute the population values for a given sequence (in QASM format) of
        operations.

        Parameters
        ----------
        sequences: dict list
            A list of control pulses/gates to perform on the device in QASM format.

        psi_init: tf.Tensor
            A tensor containing the initial statevector

        Returns
        -------
        list
            A list of populations

        """
        model = self.pmap.model
        if psi_init is None:
            psi_init = model.get_init_state()
        self.psi_init = psi_init
        populations = []
        for sequence in sequences:
            psi_t = copy.deepcopy(self.psi_init)
            for gate in sequence:
                psi_t = tf.matmul(self.lookup_gate(**gate), psi_t)

            pops = self.populations(psi_t, model.lindbladian)
            populations.append(pops)
        return populations

    def lookup_gate(self, name, qubits, params=None) -> tf.constant:
        """
        Returns a fixed operation or a parametric virtual Z gate. To be extended to
        general parametric gates.
        """
        if name == "VZ":
            gate = tf.constant(self.get_VZ(qubits, params))
        else:
            gate = self.propagators[name + str(qubits)]
        return gate

    def get_VZ(self, target, params):
        """
        Returns the appropriate Z-rotation.
        """
        dims = self.pmap.model.dims
        return perfect_single_q_parametric_gate("Z", target[0], params[0], dims)

    def process(self, populations, labels=None):
        """
        Apply a readout procedure to a population vector. Very specialized
        at the moment.

        Parameters
        ----------
        populations: list
            List of populations from evaluating.

        labels: list
            List of state labels specifying a subspace.

        Returns
        -------
        list
            A list of processed populations.

        """
        model = self.pmap.model
        populations_final = []
        populations_no_rescale = []
        for pops in populations:
            # TODO: Loop over all model.tasks in a general fashion
            # TODO: Selecting states by label in the case of computational space
            if "conf_matrix" in model.tasks:
                pops = model.tasks["conf_matrix"].confuse(pops)
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        pops_select += pops[model.comp_state_labels.index(label)]
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            else:
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        try:
                            pops_select += pops[model.state_labels.index(label)]
                        except ValueError:
                            raise Exception(
                                f"C3:ERROR:State {label} not defined. Available are:\n"
                                f"{model.state_labels}"
                            )
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            if "meas_rescale" in model.tasks:
                populations_no_rescale.append(pops)
                pops = model.tasks["meas_rescale"].rescale(pops)
            populations_final.append(pops)
        return populations_final, populations_no_rescale

    def get_perfect_gates(self, gate_keys: list = None) -> Dict[str, np.ndarray]:
        """Return a perfect gateset for the gate_keys.

        Parameters
        ----------
        gate_keys: list
            (Optional) List of gates to evaluate.

        Returns
        -------
        Dict[str, np.array]
            A dictionary of gate names and np.array representation
            of the corresponding unitary

        Raises
        ------
        Exception
            Raise general exception for undefined gate
        """
        instructions = self.pmap.instructions
        gates = {}
        dims = self.pmap.model.dims
        if gate_keys is None:
            gate_keys = instructions.keys()  # type: ignore
        for gate in gate_keys:
            gates[gate] = instructions[gate].get_ideal_gate(dims)

        # TODO parametric gates

        return gates

    def compute_propagators(self):
        """
        Compute the unitary representation of operations. If no operations are
        specified in self.opt_gates the complete gateset is computed.

        Returns
        -------
        dict
            A dictionary of gate names and their unitary representation.
        """
        model = self.pmap.model
        generator = self.pmap.generator
        instructions = self.pmap.instructions
        propagators = {}
        partial_propagators = {}
        gate_ids = self.opt_gates
        if gate_ids is None:
            gate_ids = instructions.keys()

        self.set_prop_method(self.prop_method)

        for gate in gate_ids:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )

            model.controllability = self.use_control_fields
            steps = int((instr.t_end - instr.t_start) * self.sim_res)
            result = self.propagation(
                model,
                generator,
                instr,
                self.folding_stack[steps],
                self.propagate_batch_size,
            )
            U = result["U"]
            dUs = result["dUs"]
            self.ts = result["ts"]
            if model.use_FR:
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    # TODO calculate properly the average frequency that each qubit sees
                    offset = tf.constant(0.0, tf.float64)
                    for ctrl in ctrls.values():
                        if "freq_offset" in ctrl.params.keys():
                            if ctrl.params["amp"] != 0.0:
                                offset = ctrl.params["freq_offset"].get_value()
                    freqs[line] = tf.cast(
                        ctrls["carrier"].params["freq"].get_value() + offset,
                        tf.complex128,
                    )
                    framechanges[line] = tf.cast(
                        ctrls["carrier"].params["framechange"].get_value(),
                        tf.complex128,
                    )
                t_final = tf.constant(instr.t_end - instr.t_start, dtype=tf.complex128)
                FR = model.get_Frame_Rotation(t_final, freqs, framechanges)
                if model.lindbladian:
                    SFR = tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            if model.dephasing_strength != 0.0:
                if not model.lindbladian:
                    raise ValueError("Dephasing can only be added when lindblad is on.")
                else:
                    amps = {}
                    for line, ctrls in instr.comps.items():
                        amp, sum = generator.devices["awg"].get_average_amp()
                        amps[line] = tf.cast(amp, tf.complex128)
                    t_final = tf.constant(
                        instr.t_end - instr.t_start, dtype=tf.complex128
                    )
                    dephasing_channel = model.get_dephasing_channel(t_final, amps)
                    U = tf.matmul(dephasing_channel, U)
            propagators[gate] = U
            partial_propagators[gate] = dUs

        # TODO we might want to move storing of the propagators to the instruction object
        if self.overwrite_propagators:
            self.propagators = propagators
            self.partial_propagators = partial_propagators
        else:
            self.propagators.update(propagators)
            self.partial_propagators.update(partial_propagators)
        self.compute_propagators_timestamp = time.time()
        return propagators

    def set_opt_gates(self, gates):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        gates: Identifiers of the gates of interest. Can contain duplicates.

        """
        if type(gates) is str:
            gates = [gates]
        self.opt_gates = gates

    def set_opt_gates_seq(self, seqs):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        seqs: Identifiers of the sequences of interest. Can contain duplicates.

        """
        self.opt_gates = list(set(itertools.chain.from_iterable(seqs)))

    def set_enable_store_unitaries(self, flag, logdir, exist_ok=False):
        """
        Saving of unitary propagators.

        Parameters
        ----------
        flag: boolean
            Enable or disable saving.
        logdir: str
            File path location for the resulting unitaries.
        """
        self.enable_store_unitaries = flag
        self.logdir = logdir
        if self.enable_store_unitaries:
            os.makedirs(self.logdir + "unitaries/", exist_ok=exist_ok)
            self.store_unitaries_counter = 0

    def store_Udict(self, goal):
        """
        Save unitary as text and pickle.

        Parameter
        ---------
        goal: tf.float64
            Value of the goal function, if used.

        """
        folder = (
            self.logdir
            + "unitaries/eval_"
            + str(self.store_unitaries_counter)
            + "_"
            + str(goal)
            + "/"
        )
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(folder + "Us.pickle", "wb+") as file:
            pickle.dump(self.propagators, file)
        for key, value in self.propagators.items():
            # Windows is not able to parse ":" as file path
            np.savetxt(folder + key.replace(":", ".") + ".txt", value)

    def populations(self, state, lindbladian):
        """
        Compute populations from a state or density vector.

        Parameters
        ----------
        state: tf.Tensor
            State or densitiy vector.
        lindbladian: boolean
            Specify if conversion to density matrix is needed.

        Returns
        -------
        tf.Tensor
            Vector of populations.
        """
        if lindbladian:
            rho = tf_vec_to_dm(state)
            pops = tf.math.real(tf.linalg.diag_part(rho))
            return tf.reshape(pops, shape=[pops.shape[0], 1])
        else:
            return tf.abs(state) ** 2

    def expect_oper(self, state, lindbladian, oper):
        if lindbladian:
            rho = tf_vec_to_dm(state)
        else:
            rho = tf_state_to_dm(state)
        trace = np.trace(np.matmul(rho, oper))
        return [[np.real(trace)]]  # ,[np.imag(trace)]]

    def compute_states(self, solver="rk4", step_function="schrodinger", prop_method="ode_solver"):
        """
        Use a state solver to compute the trajectory of the system.

        Returns
        -------
        List[tf.tensor]
            List of states of the system from simulation.

        """

        model = self.pmap.model
        generator = self.pmap.generator
        instructions = self.pmap.instructions

        init_state = self.pmap.model.get_init_state()
        if step_function == "von_neumann":
            init_state = tf_state_to_dm(init_state)
        ts_init = tf.constant(0.0, dtype=tf.complex128)

        state_list = tf.expand_dims(init_state, 0)
        ts_list = [ts_init]

        sequence = self.opt_gates

        self.set_prop_method(prop_method)

        for gate in sequence:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )
            result = self.propagation(
                model,
                generator,
                instr,
                init_state,
                solver=solver,
                step_function=step_function,
            )
            state_list = tf.concat([state_list, result["states"]], 0)
            ts_list = tf.concat([ts_list, tf.add(result["ts"], ts_init)], 0)
            init_state = result["states"][-1]
            ts_init = result["ts"][-1]

        return {"states": state_list, "ts": ts_list}

    def compute_final_state(self, solver="rk4", step_function="schrodinger"):
        """
        Solve the Lindblad master equation by integrating the differential
        equation of the density matrix

        Returns
        -------
        List
           Final state after time evolution.
        """

        model = self.pmap.model
        generator = self.pmap.generator
        instructions = self.pmap.instructions

        init_state = self.pmap.model.get_init_state()
        if step_function == "von_neumann":
            init_state = tf_state_to_dm(init_state)

        sequence = self.opt_gates
        self.set_prop_method("ode_solver_final_state")

        for gate in sequence:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )
            result = self.propagation(
                model,
                generator,
                instr,
                init_state,
                solver=solver,
                step_function=step_function,
            )
            init_state = result["states"]
            ts = result["ts"]

        return {"states": result["states"], "ts": ts[-1]}

    def solve_stochastic_ode(
        self, 
        init_state, 
        sequence, 
        Num_shots, 
        enable_vec_map=False, 
        batch_size=None,
        solver="rk4"
    ):
        """
        Solve the Lindblad master equation by simulating the stochastic
        schrodinger equation.

        Returns
        -------
        List
            A List of states for the time evolution.
        """

        model = self.pmap.model
        if model.lindbladian:
            raise Exception(
                "model.lindbladian is True."
                + "This method uses state vectors instead of density matrices."
            )

        model.controllability = self.use_control_fields

        self.set_prop_method("stochastic_schrodinger_rk4")

        self.sequence = sequence
        self.init_state = init_state
        self.solver = solver

        # TODO - Multiply factors of gamma to the collapse operators
        
        N_sub = len(model.subsystems)

        collapse_ops = tf.TensorArray(
                        tf.complex128,
                        size=N_sub,
                        dynamic_size=False, 
                        infer_shape=False
        )
        counter = 0
        for key in model.subsystems:
            Ls = model.subsystems[key].Ls
            collapse_ops = collapse_ops.write(counter, Ls)
            counter += 1

        collapse_ops = collapse_ops.stack()
        self.collapse_ops = collapse_ops


        L_dag_L = []
        counter = 0
        for key in model.subsystems:
            cols = [
                tf.matmul(
                tf.transpose(collapse_ops[counter][0], conjugate=True),
                collapse_ops[counter][0]
                ),
                tf.matmul(
                tf.transpose(collapse_ops[counter][1], conjugate=True),
                collapse_ops[counter][1]
                ),
                tf.matmul(
                tf.transpose(collapse_ops[counter][2], conjugate=True),
                collapse_ops[counter][2]
                )
            ]
            L_dag_L.append(cols)
            counter += 1
        self.L_dag_L = L_dag_L


        #calculating pulse lengths
        res = self.pmap.generator.devices["LO"].resolution
        instructions = self.pmap.instructions
        ts_len ={}
        for gate in sequence:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )

            ts_len[gate] = int(instr.t_end * res)

        self.ts_len = ts_len
        dt = 1/res

        plist_list = []
        for i in range(Num_shots):
            counter = 0
            for gate in sequence:
                plist = self.precompute_dissipation_probs(
                                    model, 
                                    ts_len[gate], 
                                    dt, 
                                    init_state, 
                                    collapse_ops
                )
                if counter == 0:
                    plist_list.append(plist)
                else:
                    plist_list[-1] = tf.concat([plist_list[-1], plist], 2)
                counter += 1

        plist_list = tf.convert_to_tensor(plist_list, dtype=tf.complex128)

        single_stochastic_run_tf = tf.function(self.single_stochastic_run)
        if not enable_vec_map:
            psi_shots = []
            for num in range(Num_shots):
                print(f"Running shot {num}")
                psi_list, ts_list = single_stochastic_run_tf(plist_list[num])
                psi_shots.append(psi_list)
            psi_shots = tf.convert_to_tensor(psi_shots, dtype=tf.complex128)
            ts_list = tf.convert_to_tensor(ts_list, dtype=tf.complex128)

        elif enable_vec_map and (batch_size != None):
            Num_batches = int(tf.math.ceil(Num_shots/batch_size))
            self.Num_batches = Num_batches
            self.batch_size = batch_size
            psi_shots, ts_list = self.batch_propagate_sde(plist_list)
        else:
            x = tf.convert_to_tensor(plist_list, dtype=tf.complex128)
            psi_shots, ts_list = tf.vectorized_map(single_stochastic_run_tf, x)

        if tf.reduce_any(tf.math.is_nan(tf.abs(psi_shots))):
            print("Some states are NaN.")
        # TODO - Add Frame rotation and dephasing strength
        return {"states": psi_shots, "ts": ts_list}
    
    def batch_propagate_sde(
                self, 
                plist_tensor: tf.TensorArray, 
    ):
        batch_array = tf.TensorArray(
            tf.complex128, size=self.Num_batches, dynamic_size=False, infer_shape=False
        )
        batch_size = self.batch_size
        single_stochastic_run_tf = tf.function(self.single_stochastic_run)
        for i in tf.range(self.Num_batches):
            print(f"Tracing shot {i}")
            x = plist_tensor[i*batch_size: i*batch_size + batch_size]
            psi_shots, ts_list = tf.vectorized_map(single_stochastic_run_tf, x)
            batch_array = batch_array.write(i, psi_shots)
        batch_array = batch_array.concat()

        return batch_array, ts_list


    def single_stochastic_run(self, plist_seq):
        print("Tracing Single stochastic run")
        sequence = self.sequence
        init_state = self.init_state
        instructions = self.pmap.instructions
        model = self.pmap.model
        generator = self.pmap.generator
        collapse_ops = self.collapse_ops
        L_dag_L = self.L_dag_L
        ts_len = self.ts_len
        solver = self.solver

        psi_init = init_state
        ts_init = tf.constant(0.0, dtype=tf.complex128)

        psi_list = tf.expand_dims(psi_init, 0)
        ts_list = [ts_init]

        counter = 0
        ts_last = 0

        for gate in sequence:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )
            result = self.propagation(
                            model=model, 
                            gen=generator, 
                            instr=instr, 
                            collapse_ops=collapse_ops, 
                            init_state=psi_init, 
                            L_dag_L=L_dag_L, 
                            plist=plist_seq[:,:,ts_last:ts_last + ts_len[gate]],
                            solver=solver
            )
            psi_list = tf.concat([psi_list,  result["states"]], 0)
            ts_list = tf.concat([ts_list, tf.add(result["ts"], ts_init)], 0)
            psi_init = result["states"][-1]
            ts_init = result["ts"][-1]
            ts_last += ts_len[gate]
            counter += 1
        return psi_list, ts_list

    @tf.function
    def precompute_dissipation_probs(self, model, ts_len, dt, psi, col):
        # TODO - correct the probability values
        pT1 = []
        pT2 = []
        pTemp = []

        dt = tf.cast(dt, dtype=tf.float64)
        counter = 0
        for key, sub in model.subsystems.items():
            try:
                t1_val = sub.params["t1"].get_value()
                pT1.append(dt/t1_val)
                #pT1.append(tf.abs(calculate_expectation_value(psi, col[counter][0]))[0]*dt)
            except KeyError:
                raise Exception(
                    f"Error: T1 for {key} is not defined."
                )
            try:
                t2_val = sub.params["t2star"].get_value()
                pT2.append(tf.abs(calculate_expectation_value(psi, col[counter][1]))[0]*dt)
            except KeyError:
                raise Exception(
                    f"Error: T2Star for {key} is not defined."
                )

            try:
                temp_val = sub.params["temp"].get_value()
                pTemp.append(tf.abs(calculate_expectation_value(psi, col[counter][2]))[0]*dt) #TODO - check if there is a factor of Kb
            except KeyError:
                raise Exception(
                    f"Error: Temp for {key} is not defined."
                )
            counter += 1

        plists = []
        g = tf.random.get_global_generator()
        counter = 0

        
        for key in model.subsystems:
            plists.append([])
            temp1 = g.uniform(shape=[ts_len], dtype=tf.float64)
            temp2 = g.uniform(shape=[ts_len], dtype=tf.float64)
            tempt = g.uniform(shape=[ts_len], dtype=tf.float64)

            plists[counter].append(tf.math.floor((tf.math.sign(-temp1 + pT1[counter]) + 1)/2))
            plists[counter].append(tf.math.floor((tf.math.sign(-temp2 + pT2[counter]) + 1)/2))
            plists[counter].append(tf.math.floor((tf.math.sign(-tempt + pTemp[counter]) + 1)/2))

            counter += 1
        return tf.cast(plists, dtype=tf.complex128)

