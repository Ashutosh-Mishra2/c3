import numpy as np


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

    def list_parameters(self):
        par_list = []
        par_list.extend(self.model.list_parameters())
        devices = self.generator.devices
        for key in devices:
            par_list.extend(devices[key].list_parameters())
        return par_list

    def parameter_indeces(self, opt_map: list):
        par_list = self.list_parameters()
        par_indx = []
        for par_id in opt_map:
            par_indx.append(par_list.index(par_id))
        return par_indx

    def get_parameters(self, opt_map: list):
        """Return list of values and bounds of parameters in opt_map."""
        values = []
        values.append(self.model.get_parameters())
        devices = self.generator.devices
        for key in devices:
            pars = devices[key].get_parameters()
            if not (pars == []):
                values.append(pars)
        # TODO Deal with bounds correctly
        self.par_lens = [len(list) for list in values]
        par_indx = self.parameter_indeces(opt_map)
        values_flat = []
        for list in values:
            values_flat.extend(list)
        values_new = [values_flat[indx] for indx in par_indx]
        bounds = np.kron(np.array([[0.9], [1.1]]), np.array(values_new)).T
        return values_new, bounds

    def set_parameters(self, values: list, opt_map: list):
        """Set the values in the original instruction class."""
        pars, _ = self.get_parameters(self.list_parameters())
        par_indx = self.parameter_indeces(opt_map)
        indx = 0
        for par_ii in par_indx:
            pars[par_ii] = values[indx]
            indx = indx + 1
        first_ind = 0
        last_ind = first_ind + self.par_lens[0]
        params = pars[first_ind:last_ind]
        self.model.set_parameters(params)
        devs = self.generator.devices
        indx = 0
        for par in opt_map:
            if par[0] in devs.keys():
                devs[par[0]].params[par[1]] = values[indx]
            indx += 1
