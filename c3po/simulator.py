"""Simulator."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from c3po.experiment import Experiment
from c3po.control import GateSet

from c3po.tf_utils import evaluate_sequences
from c3po.tf_utils import tf_propagation
from c3po.tf_utils import tf_propagation_lind
from c3po.tf_utils import tf_matmul_left, tf_matmul_right
from c3po.tf_utils import tf_super
from c3po.qt_utils import single_length_RB


# TODO resctucture so that sim,mdl,gen,meas,gateset are all in one class (exp?)
class Simulator():
    """Simulator object."""

    def __init__(self,
                 exp: Experiment,
                 gateset: GateSet
                 ):
        self.exp = exp
        self.gateset = gateset
        self.unitaries = {}
        self.lindbladian = False

    def get_gates(
        self,
        gateset_values: list,
        gateset_opt_map: list
    ):
        gates = {}
        self.gateset.set_parameters(gateset_values, gateset_opt_map)
        # TODO allow for not passing model params
        # model_params, _ = self.model.get_values_bounds()
        for gate in self.gateset.instructions.keys():
            signal = self.exp.generator.generate_signals(
                self.gateset.instructions[gate]
            )
            U = self.propagation(signal)
            gates[gate] = U
            self.unitaries = gates
        return gates

    def propagation(self,
                    signal: dict
                    ):
        signals = []
        # This sorting ensures that signals and hks are matched
        # TODO do hks and signals matching more rigorously
        for key in sorted(signal):
            out = signal[key]
            # TODO this points to the fact that all sim_res must be the same
            ts = out["ts"]
            signals.append(out["values"])

        dt = ts[1].numpy() - ts[0].numpy()
        h0, hks = self.exp.model.get_Hamiltonians()
        if self.lindbladian:
            col_ops = self.exp.model.get_lindbladian()
            dUs = tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_propagation(h0, hks, signals, dt)
        self.dUs = dUs
        self.ts = ts
        U = tf_matmul_left(dUs)
        if hasattr(self, 'VZ'):
            if self.lindbladian:
                U = tf.matmul(tf_super(self.VZ), U)
            else:
                U = tf.matmul(self.VZ, U)
        self.U = U
        return U

    def plot_dynamics(self,
                      psi_init
                      ):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t)
        for du in dUs:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = self.populations(psi_t)
            pop_t = np.append(pop_t, pops, axis=1)
        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.append(0, ts + dt / 2)
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        fig.show()

    def populations(self, state):
        if self.lindbladian:
            diag = []
            dim = int(np.sqrt(len(state)))
            indeces = [n * dim + n for n in range(dim)]
            for indx in indeces:
                diag.append(state[indx])
            return np.abs(diag)
        else:
            return np.abs(state)**2

    def RB(self,
           psi_init,
           U_dict,
           min_length: int = 5,
           max_length: int = 100,
           num_lengths: int = 20,
           num_seqs: int = 30,
           plot_all=True,
           progress=True,
           logspace=False
           ):
        print('performing RB experiment')
        if logspace:
            lengths = np.rint(
                        np.logspace(
                            np.log10(min_length),
                            np.log10(max_length),
                            num=num_lengths
                            )
                        ).astype(int)
        else:
            lengths = np.rint(
                        np.linspace(
                            min_length,
                            max_length,
                            num=num_lengths
                            )
                        ).astype(int)
        surv_prob = []
        for L in lengths:
            if progress:
                print(L)
            seqs = single_length_RB(num_seqs, L)
            Us = evaluate_sequences(U_dict, seqs)
            pop0s = []
            for U in Us:
                pops = self.populations(tf.matmul(U, psi_init))
                pop0s.append(float(pops[0]))
            surv_prob.append(pop0s)

        def RB_fit(len, r, A, B):
            return A * r**(len) + B
        bounds = (0, 1)
        init_guess = [0.9, 0.5, 0.5]
        fitted = False
        while not fitted:
            print('lengths shape', lengths.shape)
            print('surv_prob shape', np.array(surv_prob).shape)
            try:
                means = np.mean(surv_prob, axis=1)
                stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
                solution, cov = curve_fit(RB_fit,
                                          lengths,
                                          means,
                                          sigma=stds,
                                          bounds=bounds,
                                          p0=init_guess)
                r, A, B = solution
                fitted = True
            except Exception as message:
                print(message)
                print('increasing RB length.')
                if logspace:
                    new_lengths = np.rint(
                                np.logspace(
                                    np.log10(max_length + min_length),
                                    np.log10(max_length * 2),
                                    num=num_lengths
                                    )
                                ).astype(int)
                else:
                    new_lengths = np.rint(
                                np.linspace(
                                    max_length + min_length,
                                    max_length*2,
                                    num=num_lengths
                                    )
                                ).astype(int)
                max_length = max_length * 2
                for L in new_lengths:
                    if progress:
                        print(L)
                    seqs = single_length_RB(num_seqs, L)
                    Us = evaluate_sequences(U_dict, seqs)
                    pop0s = []
                    for U in Us:
                        pops = self.populations(tf.matmul(U, psi_init))
                        pop0s.append(float(pops[0]))
                    surv_prob.append(pop0s)
                lengths = np.append(lengths, new_lengths)


        fig, ax = plt.subplots()
        if plot_all:
            ax.plot(lengths,
                    surv_prob,
                    marker='o',
                    color='red',
                    linestyle='None')
        ax.errorbar(lengths,
                    means,
                    yerr=stds,
                    color='blue',
                    marker='x',
                    linestyle='None')
        plt.title('RB results')
        plt.ylabel('Population in 0')
        plt.xlabel('# Cliffords')
        plt.ylim(0, 1)
        plt.xlim(0, lengths[-1])
        fitted = RB_fit(lengths, r, A, B)
        ax.plot(lengths, fitted)
        plt.text(0.1, 0.1,
                 'r={:.4f}, A={:.3f}, B={:.3f}'.format(r, A, B),
                 size=16,
                 transform=ax.transAxes)
        plt.show()
        return r, A, B


    def leakage_RB(self,
       psi_init,
       U_dict,
       min_length: int = 5,
       max_length: int = 100,
       num_lengths: int = 20,
       num_seqs: int = 30,
       plot_all=True,
       progress=True,
       logspace=False,
       return_plot=False,
       fig=None,
       ax=None
    ):
        print('performing Leakage RB experiment')
        if logspace:
            lengths = np.rint(
                        np.logspace(
                            np.log10(min_length),
                            np.log10(max_length),
                            num=num_lengths
                            )
                        ).astype(int)
        else:
            lengths = np.rint(
                        np.linspace(
                            min_length,
                            max_length,
                            num=num_lengths
                            )
                        ).astype(int)
        comp_surv = []
        surv_prob = []
        for L in lengths:
            if progress:
                print(L)
            seqs = single_length_RB(num_seqs, L)
            Us = evaluate_sequences(U_dict, seqs)
            pop0s = []
            pop_comps = []
            for U in Us:
                pops = self.populations(tf.matmul(U, psi_init))
                pop0s.append(float(pops[0]))
                pop_comps.append(float(pops[0])+float(pops[1]))
            surv_prob.append(pop0s)
            comp_surv.append(pop_comps)

        def RB_leakage(len, r_leak, A_leak, B_leak):
            return A_leak + B_leak * r_leak**(len)
        bounds = (0, 1)
        init_guess = [0.9, 0.5, 0.5]
        fitted = False
        while not fitted:
            print('lengths shape', lengths.shape)
            print('surv_prob shape', np.array(comp_surv).shape)
            try:
                comp_means = np.mean(comp_surv, axis=1)
                comp_stds = np.std(comp_surv, axis=1) / np.sqrt(len(comp_surv[0]))
                solution, cov = curve_fit(RB_leakage,
                                          lengths,
                                          comp_means,
                                          sigma=comp_stds,
                                          bounds=bounds,
                                          p0=init_guess)
                r_leak, A_leak, B_leak = solution
                fitted = True
            except Exception as message:
                print(message)
                print('increasing RB length.')
                if logspace:
                    new_lengths = np.rint(
                                np.logspace(
                                    np.log10(max_length + min_length),
                                    np.log10(max_length * 2),
                                    num=num_lengths
                                    )
                                ).astype(int)
                else:
                    new_lengths = np.rint(
                                np.linspace(
                                    max_length + min_length,
                                    max_length*2,
                                    num=num_lengths
                                    )
                                ).astype(int)
                max_length = max_length * 2
                for L in new_lengths:
                    if progress:
                        print(L)
                    seqs = single_length_RB(num_seqs, L)
                    Us = evaluate_sequences(U_dict, seqs)
                    pop0s = []
                    pop_comps = []
                    for U in Us:
                        pops = self.populations(tf.matmul(U, psi_init))
                        pop0s.append(float(pops[0]))
                        pop_comps.append(float(pops[0]))
                    surv_prob.append(pop0s)
                    comp_surv.append(pop_comps)
                lengths = np.append(lengths, new_lengths)


        def RB_surv(len, r, A, C):
            return A + B_leak * r_leak**(len) + C * r**(len)
        bounds = (0, 1)
        init_guess = [0.9, 0.5, 0.5]
        surv_means = np.mean(surv_prob, axis=1)
        surv_stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
        solution, cov = curve_fit(RB_surv,
                                  lengths,
                                  surv_means,
                                  sigma=surv_stds,
                                  bounds=bounds,
                                  p0=init_guess)
        r, A, C = solution

        leakage = (1-A_leak)*(1-r_leak)
        seepage = A_leak*(1-r_leak)
        fid = 0.5*(r+1-leakage)

        if fig == None:
            fig, ax = plt.subplots(2)
        if plot_all:
            ax[0].plot(lengths,
                    comp_surv,
                    marker='o',
                    color='red',
                    linestyle='None')
        ax[0].errorbar(lengths,
                    comp_means,
                    yerr=comp_stds,
                    color='blue',
                    marker='x',
                    linestyle='None')
        ax[0].set_title('RB Leakage')
        ax[0].set_ylabel('Population in comp sub')
        ax[0].set_xlabel('# Cliffords')
        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(0, lengths[-1])
        fitted = RB_leakage(lengths, r_leak, A_leak, B_leak)
        ax[0].plot(lengths, fitted)
        ax[0].text(0.1, 0.1,
                 'r_leak={:.4f}, A_leak={:.3f}, B_leak={:.3f}'.format(r_leak, A_leak, B_leak),
                 size=16,
                 transform=ax[0].transAxes)

        if plot_all:
            ax[1].plot(lengths,
                    surv_prob,
                    marker='o',
                    color='red',
                    linestyle='None')
        ax[1].errorbar(lengths,
                    surv_means,
                    yerr=surv_stds,
                    color='blue',
                    marker='x',
                    linestyle='None')
        ax[1].set_title('RB results')
        ax[1].set_ylabel('Population in 0')
        ax[1].set_xlabel('# Cliffords')
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(0, lengths[-1])
        fitted = RB_surv(lengths, r, A, C)
        ax[1].plot(lengths, fitted)
        ax[1].text(0.1, 0.1,
                 'r={:.4f}, A={:.3f}, C={:.3f}'.format(r, A, C),
                 size=16,
                 transform=ax[1].transAxes)

        if return_plot:
            return fid, leakage, seepage, fig, ax
        else:
            fig.show()
            return fid, leakage, seepage
