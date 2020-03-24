"""Experiment class that models the whole experiment."""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import c3po.utils.tf_utils as tf_utils


# TODO add case where one only wants to pass a list of quantity objects?
class Experiment:
    """
    It models all of the behaviour of the physical experiment.

    It contains boxes that perform a part of the experiment routine.

    Parameters
    ----------
    model: Model
    generator: Generator

    """

    def __init__(self, model, generator, gateset):
        self.model = model
        self.generator = generator
        self.gateset = gateset

        self.unitaries = {}
        self.dUs = {}

        components = {}
        components.update(self.model.couplings)
        components.update(self.model.subsystems)
        components.update(self.model.tasks)
        components.update(self.generator.devices)
        self.components = components

        id_list = []
        par_lens = []
        for comp in self.components.values():
            id_list.extend(comp.list_parameters())
            for par in comp.params.values():
                par_lens.append(par.length)
        self.id_list = id_list
        self.par_lens = par_lens

    def write_config(self):
        cfg = {}
        cfg['model'] = self.model.write_config()
        cfg['generator'] = self.generator.write_config()
        cfg['gateset'] = self.gateset.write_config()
        return cfg

    def get_parameters(self, opt_map=None, scaled=False):
        if opt_map is None:
            opt_map = self.id_list
        values = []
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            par = self.components[comp_id].params[par_id]
            if scaled:
                values.extend(par.get_opt_value())
            else:
                values.append(par.get_value())
        return values

    def set_parameters(self, values: list, opt_map: list, scaled=False):
        """Set the values in the original instruction class."""
        val_indx = 0
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            id_indx = self.id_list.index(id)
            par_len = self.par_lens[id_indx]
            par = self.components[comp_id].params[par_id]
            if scaled:
                par.set_opt_value(values[val_indx:val_indx+par_len])
                val_indx += par_len
            else:
                try:
                    par.set_value(values[val_indx])
                    val_indx += 1
                except ValueError:
                    print("Value out of bounds")
                    print(f"Trying to set {id} to value {values[val_indx]}")
        self.model.update_model()

    def print_parameters(self, opt_map=None):
        ret = []
        if opt_map is None:
            opt_map = self.id_list
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            par = self.components[comp_id].params[par_id]
            nice_id = f"{comp_id}-{par_id}"
            ret.append(f"{nice_id:32}: {par}\n")
        return "".join(ret)
    # THE ROLE OF THE OLD SIMULATOR AND OTHERS

    def get_Us(self, seqs):
        U_dict = self.get_gates()
        self.U_dict = U_dict
        Us = tf_utils.evaluate_sequences(U_dict, seqs)
        return Us

    # def evaluate(self, seqs):
    #     def unit_X90p(U_dict):
    #         return fidelities.unitary_infid(U_dict, 'X90p', proj=True)
    #     U_dict = self.get_gates()
    #     results = [unit_X90p(U_dict)]
    #     return results

    def evaluate(self, seqs, labels=None):
        Us = self.get_Us(seqs)
        psi_init = self.model.tasks["init_ground"].initialise(
            self.model.drift_H,
            self.model.lindbladian
        )
        self.psi_init = psi_init

        populations_final = []
        for U in Us:
            psi_final = tf.matmul(U, self.psi_init)
            pops = self.populations(
                psi_final, self.model.lindbladian
            )
            # TODO: Loop over all tasks in a general fashion
            if "conf_matrix" in self.model.tasks:
                pops = self.model.tasks["conf_matrix"].confuse(
                    pops,
                    self.model.lindbladian
                )

            if labels is not None:
                pops_select = 0
                for l in labels:
                    pops_select += pops[self.model.comp_state_labels.index(l)]
                pops = pops_select

            if "meas_rescale" in self.model.tasks:
                pops = self.model.tasks["meas_rescale"].rescale(pops)
            populations_final.append(pops)
        return populations_final

    def get_gates(self):
        gates = {}
        # TODO allow for not passing model params
        # model_params, _ = self.model.get_values_bounds()
        if "opt_gates" in self.__dict__:
            gate_keys = self.opt_gates
        else:
            gate_keys = self.gateset.instructions.keys()
        for gate in gate_keys:
            instr = self.gateset.instructions[gate]
            signal, ts = self.generator.generate_signals(instr)
            U = self.propagation(signal, ts, gate)
            if self.model.use_FR:
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    if gate == "QId":
                        offset = 0.0
                    elif "freq_offset" in ctrls['gauss'].params:
                        offset = ctrls['gauss'].params['freq_offset'].get_value()
                    else:
                        offset = 0.0

                    freqs[line] = tf.cast(
                        ctrls['carrier'].params['freq'].get_value()
                        + offset,
                        tf.complex128
                    )
                    framechanges[line] = tf.cast(
                        ctrls['carrier'].params['framechange'].get_value(),
                        tf.complex128
                    )
                t_final = tf.constant(
                    instr.t_end - instr.t_start,
                    dtype=tf.complex128
                )
                FR = self.model.get_Frame_Rotation(
                    t_final,
                    freqs,
                    framechanges
                )
                if self.model.lindbladian:
                    SFR = tf_utils.tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            if self.model.dephasing_strength != 0.0:
                if not self.model.lindbladian:
                    raise ValueError(
                        'Dephasing can only be added when lindblad is on.'
                    )
                else:
                    amps = {}
                    for line, ctrls in instr.comps.items():
                        amp, sum = self.generator.devices['awg'].get_average_amp()
                        # amp = ctrls['gauss'].params['amp'].get_value()
                        # amp = tf.constant(1.0)
                        amps[line] = tf.cast(amp, tf.complex128)
                    t_final = tf.constant(
                        instr.t_end - instr.t_start,
                        dtype=tf.complex128
                    )
                    # t_final = tf.cast(
                    #     ctrls['gauss'].params['t_final'].get_value(),
                    #     dtype=tf.complex128
                    # )
                    # t_final = tf.constant(1.0, tf.complex128)
                    dephasing_channel = self.model.get_dephasing_channel(
                        t_final,
                        amps
                    )
                    U = tf.matmul(dephasing_channel, U)
            gates[gate] = U
            self.unitaries = gates
        return gates

    def propagation(
        self,
        signal: dict,
        ts,
        gate
    ):

        h0, hctrls = self.model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            hks.append(hctrls[key])
        dt = ts[1].numpy() - ts[0].numpy()

        if self.model.lindbladian:
            col_ops = self.model.get_Lindbladians()
            dUs = tf_utils.tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_utils.tf_propagation(h0, hks, signals, dt)
        self.dUs[gate] = dUs
        self.ts = ts
        U = tf_utils.tf_matmul_left(dUs)
        self.U = U
        return U

    def set_opt_gates(self, opt_gates):
        self.opt_gates = opt_gates

    def set_enable_dynamics_plots(self, flag, logdir):
        self.enable_dynamics_plots = flag
        self.logdir = logdir
        if self.enable_dynamics_plots:
            os.mkdir(self.logdir + "dynamics/")
            self.dynamics_plot_counter = 0

    def set_enable_pules_plots(self, flag, logdir):
        self.enable_pulses_plots = flag
        self.logdir = logdir
        if self.enable_pulses_plots:
            os.mkdir(self.logdir + "pulses/")
            self.pulses_plot_counter = 0

    def plot_dynamics(self, psi_init, seq, debug=False):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t, self.model.lindbladian)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = self.populations(psi_t, self.model.lindbladian)
                pop_t = np.append(pop_t, pops, axis=1)
            if self.model.use_FR:
                instr = self.gateset.instructions[gate]
                signal, ts = self.generator.generate_signals(instr)
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    if gate == "QId":
                        offset = 0.0
                    elif "freq_offset" in ctrls['gauss'].params:
                        offset = ctrls['gauss'].params['freq_offset'].get_value()
                    else:
                        offset = 0.0

                    freqs[line] = tf.cast(
                        ctrls['carrier'].params['freq'].get_value()
                        + offset,
                        tf.complex128
                    )
                    framechanges[line] = tf.cast(
                        ctrls['carrier'].params['framechange'].get_value(),
                        tf.complex128
                    )
                t_final = tf.constant(
                    instr.t_end - instr.t_start,
                    dtype=tf.complex128
                )
                FR = self.model.get_Frame_Rotation(
                    t_final,
                    freqs,
                    framechanges
                )
                if self.model.lindbladian:
                    FR = tf_utils.tf_super(FR)
                psi_t = tf.matmul(FR, psi_t)

        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(self.model.state_labels)
        if debug:
            plt.show()
        else:
            plt.savefig(
                self.logdir+f"dynamics/eval_{self.dynamics_plot_counter}_{seq[0]}.png", dpi=300
                )

    def plot_pulses(self, instr):
        # print(instr.name)
        # print(instr.comps)
        # print(self.generator.devices)
        signal, ts = self.generator.generate_signals(instr)
        awg = self.generator.devices["awg"]
        awg_ts = awg.ts
        inphase = awg.signal["inphase"]
        quadrature = awg.signal["quadrature"]

        if not os.path.exists(self.logdir + "pulses/eval_" + str(self.pulses_plot_counter) + "/"):
            os.mkdir(self.logdir + "pulses/eval_" + str(self.pulses_plot_counter) + "/")
        os.mkdir(self.logdir + "pulses/eval_" + str(self.pulses_plot_counter) + "/" + str(instr.name) + "/")

        fig, axs = plt.subplots(1, 1)
        # ts = self.ts
        # dt = ts[1] - ts[0]
        # ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(awg_ts / 1e-9, inphase/1e-3)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Pulse amplitude[mV]')
        plt.savefig(
            self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/awg_inphase_{list(instr.comps.keys())}.png", dpi=300
        )

        fig, axs = plt.subplots(1, 1)
        axs.plot(awg_ts / 1e-9, quadrature/1e-3)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Pulse amplitude[mV]')
        plt.savefig(
            self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/awg_quadrature_{list(instr.comps.keys())}.png", dpi=300
        )

        dac = self.generator.devices["dac"]
        dac_ts = dac.ts
        inphase = dac.signal["inphase"]
        quadrature = dac.signal["quadrature"]

        fig, axs = plt.subplots(1, 1)
        # ts = self.ts
        # dt = ts[1] - ts[0]
        # ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(dac_ts / 1e-9, inphase/1e-3)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Pulse amplitude[mV]')
        plt.savefig(
            self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/dac_inphase_{list(instr.comps.keys())}.png", dpi=300
        )

        fig, axs = plt.subplots(1, 1)
        axs.plot(dac_ts / 1e-9, quadrature/1e-3)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Pulse amplitude[mV]')
        plt.savefig(
            self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/dac_quadrature_{list(instr.comps.keys())}.png", dpi=300
        )

        if "resp" in self.generator.devices:
            resp = self.generator.devices["resp"]
            resp_ts = dac_ts
            inphase = resp.signal["inphase"]
            quadrature = resp.signal["quadrature"]

            fig, axs = plt.subplots(1, 1)
            # ts = self.ts
            # dt = ts[1] - ts[0]
            # ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
            axs.plot(resp_ts / 1e-9, inphase/1e-3)
            axs.grid()
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('Pulse amplitude[mV]')
            plt.savefig(
                self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/resp_inphase_{list(instr.comps.keys())}.png", dpi=300
            )

            fig, axs = plt.subplots(1, 1)
            axs.plot(resp_ts / 1e-9, quadrature/1e-3)
            axs.grid()
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('Pulse amplitude[mV]')
            plt.savefig(
                self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/resp_quadrature_{list(instr.comps.keys())}.png", dpi=300
            )

        fig, axs = plt.subplots(1, 1)
        # ts = self.ts
        # dt = ts[1] - ts[0]
        # ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, signal["d1"]["values"])
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('signal')
        plt.savefig(
            self.logdir+f"pulses/eval_{self.pulses_plot_counter}/{instr.name}/final_{list(instr.comps.keys())}.png", dpi=300
        )


    def populations(self, state, lindbladian):
        if lindbladian:
            rho = tf_utils.tf_vec_to_dm(state)
            pops = tf.math.real(tf.linalg.diag_part(rho))
            return tf.reshape(pops, shape=[pops.shape[0], 1])
        else:
            return tf.abs(state)**2
