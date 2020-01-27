"""Experiment class that models the whole experiment."""

import types
import copy
import numpy as np
import tensorflow as tf
from c3po.component import C3obj


class Experiment:
    """
    It models all of the behaviour of the physical experiment.

    It contains boxes that perform a part of the experiment routine.

    Parameters
    ----------
    model: Model
    generator: Generator

    """

    def __init__(self, model, generator):
        self.model = model
        self.generator = generator

        components = {}
        components.update(self.model.line_comps_dict)
        components.update(self.model.phys_comps_dict)
        components.update(self.generator.devices)
        self.components = components

        id_list = []
        for key in self.components:
            id_list.extend(self.components[key].list_parameters())
        self.id_list = id_list

    def write_config(self):
        cfg = {}
        cfg = copy.deepcopy(self.__dict__)
        for key in cfg:
            if key == 'model':
                cfg[key] = self.model.write_config()
            elif key == 'generator':
                cfg[key] = self.generator.write_config()
        return cfg

    def get_parameters(self, opt_map=None, scaled=False):
        """Return list of values and bounds of parameters in opt_map."""
        if opt_map is None:
            opt_map = self.id_list
        values = []
        par_lens = []
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            par = self.components[comp_id].get_parameter(par_id, scaled)
            par_lens.append(par.size)
            values.extend(par.flatten())
        self.par_lens = par_lens
        return values

    def set_parameters(self, values: list, opt_map: list):
        """Set the values in the original instruction class."""
        val_indx = 0
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            id_indx = self.id_list.index(id)
            par_len = self.par_lens[id_indx]
            self.components[comp_id].set_parameter(
                par_id,
                values[val_indx:val_indx+par_len]
            )

    def print_parameters(self, opt_map=None):
        if opt_map is None:
            opt_map = self.list_parameters()
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            self.components[comp_id].print_parameter(par_id)


class Measurement:
    """
    It models all of the behaviour of the measurement process.

    It includes initialization and readout errors.

    Parameters
    ----------
    Tasks: list

    """

    def __init__(self, tasks):
        self.tasks = {}
        for task in tasks:
            self.tasks[task.name] = task

    def measure(self, U_dict):
        with tf.name_scope('Measurement Tasks'):
            init_ground = self.devices["init_ground"]
            fidelity = self.devices["fidelity"]
            meas_err = self.devices["meas_err"]
            evolution = self.devices["evolution"]
            psi_init = init_ground.initialise()
            psi_final = evolution.evolve(U_dict, psi_init)
            measured = meas_err.measure(psi_final)
            fid = fidelity.process(measured)
        return fid


class Task(C3obj):
    """Task that is part of the measurement setup."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params = {}

    def get_parameters(self):
        params = []
        for key in sorted(self.params.keys()):
            params.append(self.params[key])
        return params

    def set_parameters(self, values):
        idx = 0
        for key in sorted(self.params.keys()):
            self.params[key] = values[idx]
            idx += 1

    def list_parameters(self):
        par_list = []
        for par_key in sorted(self.params.keys()):
            par_id = (self.name, par_key)
            par_list.append(par_id)
        return par_list


class InitialiseGround(Task):
    """Initialise the ground state with a given thermal distribution."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            temp: np.float64 = 0.0
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params['temp'] = temp

    def initialise(self):
        # init_state = thermal population with self.temp
        # return init_state
        pass


class MeasureExpectationZ(Task):
    """Initialise the ground state with a given thermal distribution."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            meas_error_matrix: np.array = np.array([[1.0, 0.0], [0.0, 1.0]])
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params['meas_error'] = np.flat(meas_error_matrix)

    def measure(self):
        # init_state = thermal population with self.temp
        # return init_state
        pass


class Fidelity(Task):
    """Perform the fidelity measurement."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            fidelity_fct: types.FunctionType = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.fidelity_fct = fidelity_fct

    def process(self, measured):
        # return self.fidelity_fct(measured)
        pass


class Evolution(Task):
    """Evolve initial state using U_dict."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            evolution_fct: types.FunctionType = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.evolution_fct = evolution_fct

    def evolve(self, U_dict, psi_init):
        # return self.evolution_fct(U_dict, psi_init)
        pass
