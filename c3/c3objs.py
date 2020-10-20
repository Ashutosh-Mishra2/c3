"""Basic custom objects."""

import numpy as np
import tensorflow as tf
from c3.utils.utils import num3str


class C3obj:
    """
    Represents an abstract object with parameters. To be inherited from.

    Parameters
    ----------
    name: str
        short name that will be used as identifier
    desc: str
        longer description of the component
    comment: str
        additional information about the component
    params: dict
        Parameters in this dict can be accessed and optimized
    """
    params: dict

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " "
            ):
        self.name = name
        self.desc = desc
        self.comment = comment
        self.params = {}

    def list_parameters(self):
        """
        Returns
        -------
        list
            A list of parameters this object has.
        """
        par_ids = []
        for par_key in sorted(self.params.keys()):
            par_id = (self.name, par_key)
            par_ids.append(par_id)
        return par_ids

    def print_parameter(self, par_id):
        """
        Print a given parameter.

        Parameters
        ----------
        par_id: str
            Parameter identifier
        """
        print(self.params[par_id])


class Quantity:
    """
    Represents any physical quantity used in the model or the pulse
    speficiation. For arithmetic operations just the numeric value is used. The
    value itself is stored in an optimizer friendly way as a float between -1
    and 1. The conversion is given by
        scale (value + 1) / 2 + offset

    Parameters
    ----------
    value: np.array(np.float64) or np.float64
        value of the quantity
    min: np.array(np.float64) or np.float64
        minimum this quantity is allowed to take
    max: np.array(np.float64) or np.float64
        maximum this quantity is allowed to take
    symbol: str
        latex representation
    unit: str
        physical unit

    """

    def __init__(
        self,
        # TODO how to specify two options for type
        value,
        min,
        max,
        symbol: str = '\\alpha',
        unit: str = 'unspecified'
    ):
        value = np.array(value)
        self.offset = np.array(min)
        self.scale = np.abs(np.array(max) - np.array(min))
        # TODO this testing should happen outside
        try:
            self.set_value(value)
        except ValueError:
            raise ValueError(
                f"Value has to be within {min:.3} .. {max:.3}"
                f" but is {value:.3}."
            )
        self.symbol = symbol
        self.unit = unit
        if hasattr(value, "shape"):
            self.shape = value.shape
            self.length = int(np.prod(value.shape))
        else:
            self.shape = ()
            self.length = 1

    def __add__(self, other):
        return self.numpy() + other

    def __radd__(self, other):
        return self.numpy() + other

    def __sub__(self, other):
        return self.numpy() - other

    def __rsub__(self, other):
        return other - self.numpy()

    def __mul__(self, other):
        return self.numpy() * other

    def __rmul__(self, other):
        return self.numpy() * other

    def __pow__(self, other):
        return self.numpy() ** other

    def __rpow__(self, other):
        return other ** self.numpy()

    def __truediv__(self, other):
        return self.numpy() / other

    def __rtruediv__(self, other):
        return other / self.numpy()

    def __str__(self):
        val = self.numpy()
        use_prefix = True
        if self.unit == "Hz 2pi":
            val = val / 2 / np.pi
        elif self.unit == "pi":
            val = val / np.pi
            use_prefix = False
        ret = ""
        for q in num3str(val, use_prefix):
            ret += q + self.unit + " "
        return ret

    def numpy(self):
        """
        Return the value of this quantity as numpy.
        """
        return self.scale * (self.value.numpy() + 1) / 2 + self.offset

    def get_value(self, val=None):
        """
        Return the value of this quantity as tensorflow.

        Parameters
        ----------
        val : tf.float64
            Optionaly give an optimizer friendly value between -1 and 1 to
            convert to physical scale.
        """
        if val is None:
            val = self.value
        return self.scale * (val + 1) / 2 + self.offset

    def set_value(self, val):
        """ Set the value of this quantity as tensorflow. Value needs to be
        within specified min and max."""
        # setting can be numpyish
        tmp = 2 * (np.array(val) - self.offset) / self.scale - 1
        if np.any(tmp < -1) or np.any(tmp > 1):
            # TODO choose which error to raise
            # raise Exception(f"Value {val} out of bounds for quantity.")
            raise ValueError
            # TODO if we want we can extend bounds when force flag is given
        else:
            self.value = tf.constant(tmp, dtype=tf.float64)

    def get_opt_value(self):
        """ Get an optimizer friendly representation of the value."""
        return self.value.numpy().flatten()

    def set_opt_value(self, val):
        """ Set value optimizer friendly.

        Parameters
        ----------
        val : tf.float64
            Tensorflow number that will be mapped to a value between -1 and 1.
        """
        self.value = tf.acos(tf.cos(
            (tf.reshape(val, self.shape) + 1) * np.pi / 2
        )) / np.pi * 2 - 1


class ParameterMap:
    """
    Collects information about control and model parameters and provides different representations depending on use.
    """

    def __init__(
        self,
        instructions: list,
        generator,
        model
    ):
        self.__instructions = {}
        for instr in instructions:
            self.__instructions[instr.name] = instr

        # Collecting model components
        components = {}
        if model:
            self.__model = model
            components.update(model.couplings)
            components.update(model.subsystems)
            components.update(model.tasks)
        if generator:
            components.update(generator.devices)
        self.__components = components

        par_lens = {}
        pars = {}
        # Initializing model parameters
        for comp in self.__components.values():
            for par_name, par_value in comp.params.items():
                par_id = (comp.name, par_name)
                par_lens[par_id] = par_value.length
                pars[par_id] = par_value

        # Initializing control parameters
        for gate in self.__instructions.keys():
            instr = self.__instructions[gate]
            for chan in instr.comps.keys():
                for comp in instr.comps[chan]:
                    for par_name, par_value in instr.comps[chan][comp].params.items():
                        par_id = (gate, chan, comp, par_name)
                        par_lens[par_id] = par_value.length
                        pars[par_id] = par_value

        self.__par_lens = par_lens
        self.__pars = pars

    def write_config(self):
        cfg = {}
        for instr in self.__instructions:
            cfg[instr] = self.__instructions[instr].write_config()
        return cfg

    def get_full_params(self):
        return self.__pars

    def get_parameters(self, opt_map=None):
        """
        Return the current parameters.

        Parameters
        ----------
        opt_map: tuple
            Hierarchical identifier for parameters.

        Returns
        -------
        list of Quantity

        """
        values = []
        for equiv_ids in opt_map:
            try:
                values.append(self.__pars[equiv_ids[0]])
            except KeyError:
                for id in self.__pars.keys():
                    if id[0] == equiv_ids[0][0]:
                        print(f"Found {id[0]}.")
                raise Exception(f"C3:ERROR:Parameter {equiv_ids[0]} not defined.")
        return values

    def set_parameters(self, values: list, opt_map: list):
        """Set the values in the original instruction class.

        Parameters
        ----------
        values: list
            List of parameter values. Can be nested, if a parameter is matrix valued.
        opt_map: list
            Corresponding identifiers for the parameter values.

        """
        val_indx = 0
        for equiv_ids in opt_map:
            for id in equiv_ids:
                try:
                    par = self.__pars[id]
                except ValueError:
                    raise Exception(f"C3:ERROR:{id} not defined.")
                try:
                    par.set_value(values[val_indx])
                    val_indx += 1
                except ValueError:
                    raise Exception(
                        f"C3:ERROR:Trying to set {'-'.join(id)} to value {values[val_indx]} "
                        f"but has to be within {par.offset:.3} .. {(par.offset + par.scale):.3}."
                    )

    def get_parameters_scaled(self, opt_map=None):
        """
        Return the current parameters.

        Parameters
        ----------
        opt_map: tuple
            Hierarchical identifier for parameters.

        Returns
        -------
        list of Quantity

        """
        values = []
        for equiv_ids in opt_map:
            par = self.__pars[equiv_ids[0]]
            values.append(par.get_opt_value())
        return values

    def set_parameters_scaled(self, values: list, opt_map: list):
        """Set the values in the original instruction class.

        Parameters
        ----------
        values: list
            List of parameter values. Matrix valued parameters need to be flattened.
        opt_map: list
            Corresponding identifiers for the parameter values.

        """
        val_indx = 0
        for equiv_ids in opt_map:
            for id in equiv_ids:
                par = self.__pars[id]
                par_len = self.__par_lens[id]
                par.set_opt_value(values[val_indx:val_indx+par_len])
                val_indx += par_len

    def print_parameters(self, opt_map=None):
        """
        Return a multi-line human-readable string of the parameter names and
        current values.

        Parameters
        ----------
        opt_map: list
            Optionally use only the specified parameters.

        Returns
        -------
        str
            Parameters and their values
        """
        ret = []

        for par_id, par in self.__pars.items():
            nice_id = "-".join(par_id)
            ret.append(f"{nice_id:38}: {par}\n")

        return "".join(ret)
