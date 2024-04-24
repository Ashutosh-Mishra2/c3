"""Library of fidelity functions."""

import numpy as np
import tensorflow as tf
from typing import List, Dict

from scipy.optimize import curve_fit

from c3.signal.gates import Instruction
from c3.utils.tf_utils import (
    tf_ave,
    tf_super,
    tf_abs,
    tf_ketket_fid,
    tf_superoper_unitary_overlap,
    tf_unitary_overlap,
    tf_project_to_comp,
    tf_dm_to_vec,
    tf_average_fidelity,
    tf_superoper_average_fidelity,
    tf_state_to_dm,
    tf_vec_to_dm,
    partial_trace_two_systems,
)

from c3.libraries.propagation import evaluate_sequences

from c3.utils.qt_utils import (
    basis,
    perfect_cliffords,
    cliffords_decomp,
    cliffords_decomp_xId,
    single_length_RB,
    cliffords_string,
)

state_providers = dict()
unitary_providers = dict()
set_providers = dict()
super_providers = dict()

# Compatibility for legacy, i.e. not sorted yet
fidelities = dict()


def fid_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    fidelities[str(func.__name__)] = func
    return func


def state_deco(func):
    """
    Decorator for making registry of functions
    """
    state_providers[str(func.__name__)] = func
    return func


def unitary_deco(func):
    """
    Decorator for making registry of functions
    """
    unitary_providers[str(func.__name__)] = func
    return func


def set_deco(func):
    """
    Decorator for making registry of functions
    """
    set_providers[str(func.__name__)] = func
    return func


def open_system_deco(func):
    """
    Decorator for making registry of functions
    """
    super_providers[str(func.__name__)] = func
    return func


@fid_reg_deco
@state_deco
def state_transfer_infid_set(
    propagators: dict, instructions: dict, index, dims, psi_0, n_eval=-1, proj=True
):
    """
    Mean state transfer infidelity.

    Parameters
    ----------
    propagators : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0 : tf.Tensor
        Initial state of the device
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        State infidelity, averaged over the gates in propagators
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        infid = state_transfer_infid(perfect_gate, propagator, index, dims, psi_0)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
@state_deco
def state_transfer_infid(ideal: np.ndarray, actual: tf.constant, index, dims, psi_0):
    """
    Single gate state transfer infidelity. The dimensions of psi_0 and ideal need to be
    compatible and index and dims need to project actual to these same dimensions.

    Parameters
    ----------
    ideal: np.ndarray
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0: tf.Tensor
        Initial state

    Returns
    -------
    tf.float
        State infidelity for the selected gate

    """
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index)
    psi_ideal = tf.matmul(ideal, psi_0)
    psi_actual = tf.matmul(actual_comp, psi_0)
    overlap = tf_ketket_fid(psi_ideal, psi_actual)
    infid = 1 - overlap
    return infid


@fid_reg_deco
@unitary_deco
def unitary_infid(
    ideal: np.ndarray, actual: tf.Tensor, index: List[int] = None, dims=None
) -> tf.Tensor:
    """
    Unitary overlap between ideal and actually performed gate.

    Parameters
    ----------
    ideal : np.ndarray
        Ideal or goal unitary representation of the gate.
    actual : np.ndarray
        Actual, physical unitary representation of the gate.
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    gate : str
        One of the keys of propagators, selects the gate to be evaluated
    dims : list
        List of dimensions of qubits

    Returns
    -------
    tf.float
        Unitary fidelity.
    """
    if index is None:
        index = list(range(len(dims)))
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index)
    fid_lvls = 2 ** len(index)
    infid = 1 - tf_unitary_overlap(actual_comp, ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
@unitary_deco
@set_deco
def unitary_infid_set(propagators: dict, instructions: dict, index, dims, n_eval=-1):
    """
    Mean unitary overlap between ideal and actually performed gate for the gates in
    propagators.

    Parameters
    ----------
    propagators : dict
        Contains actual unitary representations of the gates, resulting from physical
        simulation
    instructions : dict
        Contains the perfect unitary representations of the gates, identified by a key.
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    n_eval : int
        Number of evaluation

    Returns
    -------
    tf.float
        Unitary fidelity.
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims, index)
        infid = unitary_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
@open_system_deco
def lindbladian_unitary_infid(
    ideal: np.ndarray, actual: tf.constant, index=[0], dims=[2]
) -> tf.constant:
    """
    Variant of the unitary fidelity for the Lindbladian propagator.

    Parameters
    ----------
    ideal: np.ndarray
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits

    Returns
    -------
    tf.float
        Overlap fidelity for the Lindblad propagator.
    """
    U_ideal = tf_super(ideal)
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index, to_super=True)
    fid_lvls = 2 ** len(index)
    infid = 1 - tf_superoper_unitary_overlap(actual_comp, U_ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
@open_system_deco
@set_deco
def lindbladian_unitary_infid_set(
    propagators: dict, instructions: Dict[str, Instruction], index, dims, n_eval
):
    """
    Variant of the mean unitary fidelity for the Lindbladian propagator.

    Parameters
    ----------
    propagators : dict
        Contains actual unitary representations of the gates, resulting from physical
        simulation
    instructions : dict
        Contains the perfect unitary representations of the gates, identified by a key.
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    n_eval : int
        Number of evaluation

    Returns
    -------
    tf.float
        Mean overlap fidelity for the Lindblad propagator for all gates in propagators.
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        infid = lindbladian_unitary_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
@open_system_deco
def average_infid(
    ideal: np.ndarray, actual: tf.Tensor, index: List[int] = [0], dims=[2]
) -> tf.constant:
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.

    Parameters
    ----------
    ideal: np.ndarray
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    """
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index)
    fid_lvls = [2] * len(index)
    infid = 1 - tf_average_fidelity(actual_comp, ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
@open_system_deco
@set_deco
def average_infid_set(
    propagators: dict, instructions: dict, index: List[int], dims, n_eval=-1
):
    """
    Mean average fidelity over all gates in propagators.

    Parameters
    ----------
    propagators : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float64
        Mean average fidelity
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims, index)
        infid = average_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
@open_system_deco
@set_deco
def average_infid_seq(propagators: dict, instructions: dict, index, dims, n_eval=-1):
    """
    Average sequence fidelity over all gates in propagators.

    Parameters
    ----------
    propagators : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float64
        Mean average fidelity
    """
    fid = 1
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        fid *= 1 - average_infid(perfect_gate, propagator, index, dims)
    return 1 - fid


@fid_reg_deco
@open_system_deco
def lindbladian_average_infid(
    ideal: np.ndarray, actual: tf.constant, index=[0], dims=[2]
) -> tf.constant:
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.

    Parameters
    ----------
    ideal: np.ndarray
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    """
    U_ideal = tf_super(ideal)
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index, to_super=True)
    infid = 1 - tf_superoper_average_fidelity(actual_comp, U_ideal, lvls=dims)
    return infid


@fid_reg_deco
@open_system_deco
@set_deco
def lindbladian_average_infid_set(
    propagators: dict, instructions: Dict[str, Instruction], index, dims, n_eval
):
    """
    Mean average fidelity over all gates in propagators.

    Parameters
    ----------
    propagators : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float64
        Mean average fidelity
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        infid = lindbladian_average_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def epc_analytical(propagators: dict, index, dims, proj: bool, cliffords=False):
    # TODO check this work with new index and dims (double-check)
    num_gates = len(dims)
    if cliffords:
        real_cliffords = evaluate_sequences(
            propagators, [[C] for C in cliffords_string]
        )
    elif num_gates == 1:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp)
    elif num_gates == 2:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp_xId)
    ideal_cliffords = perfect_cliffords(lvls=[2] * num_gates, num_gates=num_gates)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        C_ideal = tf.constant(ideal_cliffords[C_indx], dtype=tf.complex128)
        ave_fid = tf_average_fidelity(C_real, C_ideal, lvls=dims)
        fids.append(ave_fid)
    infid = 1 - tf_ave(fids)
    return infid


@fid_reg_deco
def lindbladian_epc_analytical(
    propagators: dict, index, dims, proj: bool, cliffords=False
):
    num_gates = len(dims)
    if cliffords:
        real_cliffords = evaluate_sequences(
            propagators, [[C] for C in cliffords_string]
        )
    elif num_gates == 1:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp)
    elif num_gates == 2:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp_xId)
    ideal_cliffords = perfect_cliffords(lvls=[2] * num_gates, num_gates=num_gates)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        C_ideal = tf_super(tf.constant(ideal_cliffords[C_indx], dtype=tf.complex128))
        ave_fid = tf_superoper_average_fidelity(C_real, C_ideal, lvls=dims)
        fids.append(ave_fid)
    infid = 1 - tf_ave(fids)
    return infid


@fid_reg_deco
def populations(state, lindbladian):
    if lindbladian:
        diag = []
        dim = int(np.sqrt(len(state)))
        indeces = [n * dim + n for n in range(dim)]
        for indx in indeces:
            diag.append(state[indx])
        return np.abs(diag)
    else:
        return np.abs(state) ** 2


@fid_reg_deco
def population(propagators: dict, lvl: int, gate: str):
    U = propagators[gate]
    lvls = U.shape[0]
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    psi_actual = tf.matmul(U, psi_0)
    return populations(psi_actual, lindbladian=False)[lvl]


def lindbladian_population(propagators: dict, lvl: int, gate: str):
    U = propagators[gate]
    lvls = int(np.sqrt(U.shape[0]))
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    dv_0 = tf_dm_to_vec(tf_state_to_dm(psi_0))
    dv_actual = tf.matmul(U, dv_0)
    return populations(dv_actual, lindbladian=True)[lvl]


@fid_reg_deco
def RB(
    propagators,
    min_length: int = 5,
    max_length: int = 500,
    num_lengths: int = 20,
    num_seqs: int = 30,
    logspace=False,
    lindbladian=False,
    padding="",
):
    gate = list(propagators.keys())[0]
    U = propagators[gate]
    dim = int(U.shape[0])
    psi_init = tf.constant(basis(dim, 0), dtype=tf.complex128)
    if logspace:
        lengths = np.rint(
            np.logspace(np.log10(min_length), np.log10(max_length), num=num_lengths)
        ).astype(int)
    else:
        lengths = np.rint(np.linspace(min_length, max_length, num=num_lengths)).astype(
            int
        )
    surv_prob = []
    for L in lengths:
        seqs = single_length_RB(num_seqs, L, padding)
        Us = evaluate_sequences(propagators, seqs)
        pop0s = []
        for U in Us:
            pops = populations(tf.matmul(U, psi_init), lindbladian)
            pop0s.append(float(pops[0]))
        surv_prob.append(pop0s)

    def RB_fit(len, r, A, B):
        return A * r ** (len) + B

    bounds = (0, 1)
    init_guess = [0.9, 0.5, 0.5]
    fitted = False
    while not fitted:
        try:
            means = np.mean(surv_prob, axis=1)
            stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
            solution, cov = curve_fit(
                RB_fit, lengths, means, sigma=stds, bounds=bounds, p0=init_guess
            )
            r, A, B = solution
            fitted = True
        except Exception as message:
            print(message)
            if logspace:
                new_lengths = np.rint(
                    np.logspace(
                        np.log10(max_length + min_length),
                        np.log10(max_length * 2),
                        num=num_lengths,
                    )
                ).astype(int)
            else:
                new_lengths = np.rint(
                    np.linspace(
                        max_length + min_length, max_length * 2, num=num_lengths
                    )
                ).astype(int)
            max_length = max_length * 2
            for L in new_lengths:
                seqs = single_length_RB(num_seqs, L, padding)
                Us = evaluate_sequences(propagators, seqs)
                pop0s = []
                for U in Us:
                    pops = populations(tf.matmul(U, psi_init), lindbladian)
                    pop0s.append(float(pops[0]))
                surv_prob.append(pop0s)
            lengths = np.append(lengths, new_lengths)
    epc = 0.5 * (1 - r)
    epg = 1 - ((1 - epc) ** (1 / 4))  # TODO: adjust to be mean length of
    return epg


@fid_reg_deco
def lindbladian_RB_left(
    propagators: dict,
    gate: str,
    index,
    dims,
    proj: bool = False,
):
    return RB(propagators, padding="left")


@fid_reg_deco
def lindbladian_RB_right(propagators: dict, gate: str, index, dims, proj: bool):
    return RB(propagators, padding="right")


@fid_reg_deco
def leakage_RB(
    propagators,
    min_length: int = 5,
    max_length: int = 500,
    num_lengths: int = 20,
    num_seqs: int = 30,
    logspace=False,
    lindbladian=False,
):
    gate = list(propagators.keys())[0]
    U = propagators[gate]
    dim = int(U.shape[0])
    psi_init = tf.constant(basis(dim, 0), dtype=tf.complex128)
    if logspace:
        lengths = np.rint(
            np.logspace(np.log10(min_length), np.log10(max_length), num=num_lengths)
        ).astype(int)
    else:
        lengths = np.rint(np.linspace(min_length, max_length, num=num_lengths)).astype(
            int
        )
    comp_surv = []
    surv_prob = []
    for L in lengths:
        seqs = single_length_RB(num_seqs, L)
        Us = evaluate_sequences(propagators, seqs)
        pop0s = []
        pop_comps = []
        for U in Us:
            pops = populations(tf.matmul(U, psi_init), lindbladian)
            pop0s.append(float(pops[0]))
            pop_comps.append(float(pops[0]) + float(pops[1]))
        surv_prob.append(pop0s)
        comp_surv.append(pop_comps)

    def RB_leakage(len, r_leak, A_leak, B_leak):
        return A_leak + B_leak * r_leak ** (len)

    bounds = (0, 1)
    init_guess = [0.9, 0.5, 0.5]
    fitted = False
    while not fitted:
        try:
            comp_means = np.mean(comp_surv, axis=1)
            comp_stds = np.std(comp_surv, axis=1) / np.sqrt(len(comp_surv[0]))
            solution, cov = curve_fit(
                RB_leakage,
                lengths,
                comp_means,
                sigma=comp_stds,
                bounds=bounds,
                p0=init_guess,
            )
            r_leak, A_leak, B_leak = solution
            fitted = True
        except Exception as message:
            print(message)
            if logspace:
                new_lengths = np.rint(
                    np.logspace(
                        np.log10(max_length + min_length),
                        np.log10(max_length * 2),
                        num=num_lengths,
                    )
                ).astype(int)
            else:
                new_lengths = np.rint(
                    np.linspace(
                        max_length + min_length, max_length * 2, num=num_lengths
                    )
                ).astype(int)
            max_length = max_length * 2
            for L in new_lengths:
                seqs = single_length_RB(num_seqs, L)
                Us = evaluate_sequences(propagators, seqs)
                pop0s = []
                pop_comps = []
                for U in Us:
                    pops = populations(tf.matmul(U, psi_init), lindbladian)
                    pop0s.append(float(pops[0]))
                    pop_comps.append(float(pops[0]))
                surv_prob.append(pop0s)
                comp_surv.append(pop_comps)
            lengths = np.append(lengths, new_lengths)

    def RB_surv(len, r, A, C):
        return A + B_leak * r_leak ** (len) + C * r ** (len)

    bounds = (0, 1)
    init_guess = [0.9, 0.5, 0.5]

    fitted = False
    while not fitted:
        try:
            surv_means = np.mean(surv_prob, axis=1)
            surv_stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
            solution, cov = curve_fit(
                RB_surv,
                lengths,
                surv_means,
                sigma=surv_stds,
                bounds=bounds,
                p0=init_guess,
            )
            r, A, C = solution
            fitted = True
        except Exception as message:
            print(message)
            if logspace:
                new_lengths = np.rint(
                    np.logspace(
                        np.log10(max_length + min_length),
                        np.log10(max_length * 2),
                        num=num_lengths,
                    )
                ).astype(int)
            else:
                new_lengths = np.rint(
                    np.linspace(
                        max_length + min_length, max_length * 2, num=num_lengths
                    )
                ).astype(int)
            max_length = max_length * 2
            for L in new_lengths:
                seqs = single_length_RB(num_seqs, L)
                Us = evaluate_sequences(propagators, seqs)
                pop0s = []
                pop_comps = []
                for U in Us:
                    pops = populations(tf.matmul(U, psi_init), lindbladian)
                    pop0s.append(float(pops[0]))
                    pop_comps.append(float(pops[0]))
                surv_prob.append(pop0s)
                comp_surv.append(pop_comps)
            lengths = np.append(lengths, new_lengths)

    leakage = (1 - A_leak) * (1 - r_leak)
    seepage = A_leak * (1 - r_leak)
    fid = 0.5 * (r + 1 - leakage)
    epc = 1 - fid
    return epc, leakage, seepage, r_leak, A_leak, B_leak, r, A, C


@fid_reg_deco
def orbit_infid(
    propagators,
    RB_number: int = 30,
    RB_length: int = 20,
    lindbladian=False,
    shots: int = None,
    seqs=None,
    noise=None,
):
    if not seqs:
        seqs = single_length_RB(RB_number=RB_number, RB_length=RB_length)
    Us = evaluate_sequences(propagators, seqs)
    infids = []
    for U in Us:
        dim = int(U.shape[0])
        psi_init = tf.constant(basis(dim, 0), dtype=tf.complex128)
        psi_actual = tf.matmul(U, psi_init)
        pop0 = tf_abs(psi_actual[0]) ** 2
        p1 = 1 - pop0
        if shots:
            vals = tf.keras.backend.random_binomial(
                [shots],
                p=p1,
                dtype=tf.float64,
            )
            # if noise:
            #     vals = vals + (np.random.randn(shots) * noise)
            infid = tf.reduce_mean(vals)
        else:
            infid = p1
            # if noise:
            #     infid = infid + (np.random.randn() * noise)
        if noise:
            infid = infid + (np.random.randn() * noise)

        infids.append(infid)
    return tf_ave(infids)


@fid_reg_deco
def IQ_plane_distance(
    propagators: dict, instructions: dict, index, dims, params, n_eval=-1
):
    infids = []
    psi_g = params["ground_state"]
    psi_e = params["excited_state"]
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]

    if lindbladian:
        psi_g = tf_dm_to_vec(psi_g)
        psi_e = tf_dm_to_vec(psi_e)

    for gate, propagator in propagators.items():
        psi_g_t = tf.matmul(propagator, psi_g)
        psi_e_t = tf.matmul(propagator, psi_e)

        if lindbladian:
            psi_g_t = tf_vec_to_dm(psi_g_t)
            psi_e_t = tf_vec_to_dm(psi_e_t)
            alpha0 = tf.linalg.trace(tf.matmul(psi_g_t, a_rotated))
            alpha1 = tf.linalg.trace(tf.matmul(psi_e_t, a_rotated))
        else:
            alpha0 = tf.matmul(
                tf.matmul(tf.transpose(psi_g_t, conjugate=True), a_rotated), psi_g_t
            )[0, 0]
            alpha1 = tf.matmul(
                tf.matmul(tf.transpose(psi_e_t, conjugate=True), a_rotated), psi_e_t
            )[0, 0]
        distance = tf.abs(alpha0 - alpha1)
        infids.append(tf.exp(-distance / d_max))

    return tf.reduce_mean(infids)


@fid_reg_deco
def state_transfer_infid_set_full(
    propagators: dict, instructions: dict, index, dims, params, proj=True, n_eval=-1
):
    """
    Mean state transfer infidelity.
    Parameters
    ----------
    propagators : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0 : tf.Tensor
        Initial state of the device
    proj : boolean
        Project to computational subspace
    Returns
    -------
    tf.float
        State infidelity, averaged over the gates in propagators
    """
    psi0 = params["psi0"]
    lindbladian = params["lindbladian"]

    if lindbladian:
        swap_propagator = tf.eye(tf.square(tf.reduce_prod(dims)), dtype=tf.complex128)
        swap_propagator_ideal = tf.eye(
            tf.square(tf.reduce_prod(dims)), dtype=tf.complex128
        )
    else:
        swap_propagator = tf.eye(tf.reduce_prod(dims), dtype=tf.complex128)
        swap_propagator_ideal = tf.eye(tf.reduce_prod(dims), dtype=tf.complex128)

    for gate, propagator in propagators.items():
        print(gate)
        swap_propagator = tf.matmul(propagator, swap_propagator)
        perfect_gate = instructions[gate].get_ideal_gate(dims, full_hilbert_space=True)
        if lindbladian:
            perfect_gate = tf_super(perfect_gate)
            psi0 = tf_dm_to_vec(psi0)
        swap_propagator_ideal = tf.matmul(perfect_gate, swap_propagator_ideal)

    infid = state_transfer_infid_full(
        swap_propagator_ideal, swap_propagator, index, dims, psi0, lindbladian
    )
    return infid


@fid_reg_deco
def state_transfer_infid_full(
    ideal: np.ndarray, actual: tf.constant, index, dims, psi_0, lindbladian
):
    """
    Single gate state transfer infidelity. The dimensions of psi_0 and ideal need to be
    compatible and index and dims need to project actual to these same dimensions.
    Parameters
    ----------
    ideal: np.array
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0: tf.Tensor
        Initial state
    Returns
    -------
    tf.float
        State infidelity for the selected gate
    """
    psi_ideal = tf.matmul(ideal, psi_0)
    psi_actual = tf.matmul(actual, psi_0)
    if lindbladian:
        psi_actual = tf_vec_to_dm(psi_actual)
        psi_ideal = tf_vec_to_dm(psi_ideal)
        overlap = tf.linalg.trace(tf.matmul(psi_ideal, psi_actual))
    else:
        overlap = tf_ketket_fid(psi_ideal, psi_actual)
    infid = 1 - overlap
    return tf.abs(infid)


@fid_reg_deco
def swap_and_readout(
    propagators: dict, instructions: dict, index, dims, params, n_eval=-1
):
    print("Calculating fidelity")
    infids = []
    psi_g = params["ground_state"]
    psi_e = params["excited_state"]
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    psi_0 = params["psi_0"]
    swap_cost = params["swap_cost"]
    lindbladian = params["lindbladian"]

    swap_cost = tf.constant(swap_cost, dtype=tf.complex128)
    infid = tf.convert_to_tensor(0.0, dtype=tf.complex128)
    if lindbladian:
        swap_propagator = tf.eye(tf.square(tf.reduce_prod(dims)), dtype=tf.complex128)
        swap_propagator_ideal = tf.eye(
            tf.square(tf.reduce_prod(dims)), dtype=tf.complex128
        )
    else:
        swap_propagator = tf.eye(tf.reduce_prod(dims), dtype=tf.complex128)
        swap_propagator_ideal = tf.eye(tf.reduce_prod(dims), dtype=tf.complex128)

    if lindbladian:
        psi_g = tf_dm_to_vec(psi_g)
        psi_e = tf_dm_to_vec(psi_e)
        psi_0 = tf_dm_to_vec(psi_0)

    for gate, propagator in propagators.items():
        print(gate)
        if "Readout" in gate:
            psi_g_t = tf.matmul(propagator, psi_g)
            psi_e_t = tf.matmul(propagator, psi_e)

            if lindbladian:
                psi_g_t = tf_vec_to_dm(psi_g_t)
                psi_e_t = tf_vec_to_dm(psi_e_t)
                alpha0 = tf.linalg.trace(tf.matmul(psi_g_t, a_rotated))
                alpha1 = tf.linalg.trace(tf.matmul(psi_e_t, a_rotated))
            else:
                alpha0 = tf.matmul(
                    tf.matmul(tf.transpose(psi_g_t, conjugate=True), a_rotated), psi_g_t
                )[0, 0]
                alpha1 = tf.matmul(
                    tf.matmul(tf.transpose(psi_e_t, conjugate=True), a_rotated), psi_e_t
                )[0, 0]

            distance = tf.abs(alpha0 - alpha1)
            iq_infid = tf.exp(-distance / d_max)
            iq_infid = tf.cast(iq_infid, dtype=tf.complex128)
            infid += iq_infid

        if "swap" in gate:
            swap_propagator = tf.matmul(propagator, swap_propagator)
            if lindbladian:
                ideal_gate = tf_super(
                    instructions[gate].get_ideal_gate(dims, full_hilbert_space=True)
                )
            else:
                ideal_gate = instructions[gate].get_ideal_gate(
                    dims, full_hilbert_space=True
                )
            swap_propagator_ideal = tf.matmul(ideal_gate, swap_propagator_ideal)

    # swap infideltiy
    if lindbladian:
        psi_swap_ideal = tf.matmul(swap_propagator_ideal, psi_0)
        psi_swap_actual = tf.matmul(swap_propagator, psi_0)

        psi_swap_ideal = tf_vec_to_dm(psi_swap_ideal)
        psi_swap_actual = tf_vec_to_dm(psi_swap_actual)

        overlap = tf.linalg.trace(tf.matmul(psi_swap_ideal, psi_swap_actual))
    else:
        psi_swap_ideal = tf.matmul(swap_propagator_ideal, psi_0)
        psi_swap_actual = tf.matmul(swap_propagator, psi_0)
        overlap = tf_ketket_fid(psi_swap_ideal, psi_swap_actual)

    swap_infid = 1 - overlap
    swap_infid = tf.cast(swap_infid, dtype=tf.complex128)
    infid += swap_cost * swap_infid

    infids.append(infid)

    return tf.abs(tf.reduce_mean(infids))


@fid_reg_deco
def state_transfer_from_states(states: tf.Tensor, index, dims, params, n_eval=-1):
    psi_0 = params["target"]

    if len(states.shape) > 2:
        overlap = calculate_state_overlap(states[-1], psi_0)
    else:
        overlap = calculate_state_overlap(states, psi_0)

    infid = 1 - overlap
    return tf.abs(infid)


def calculate_state_overlap(psi1, psi2):
    if psi1.shape[0] == psi1.shape[1]:
        return (tf.linalg.trace(tf.abs(tf.sqrt(tf.matmul(psi1, psi2))))) ** 2
    else:
        return tf_ketket_fid(psi1, psi2)[0]


def calculate_state_overlap_batch(psi_list, psi_target):
    if psi_list.shape[1] == psi_list.shape[2]:
        return (tf.linalg.trace(tf.abs(tf.sqrt(tf.matmul(psi_list, psi_target))))) ** 2
    else:
        return tf.reshape(
            tf.abs(tf.matmul(tf.linalg.adjoint(psi_list), psi_target)),
            psi_list.shape[0],
        )


def calculate_expect_value(psi, op, lindbladian):
    if lindbladian:
        expect = tf.linalg.trace(tf.matmul(psi, op))
    else:
        expect = tf.matmul(tf.matmul(tf.transpose(psi, conjugate=True), op), psi)[0, 0]
    return expect


@fid_reg_deco
def readout_ode(states: tf.Tensor, index, dims, params, n_eval=-1):
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]
    psi_g = states[0][-1]
    psi_e = states[1][-1]

    alpha0 = calculate_expect_value(psi_g, a_rotated, lindbladian)
    alpha1 = calculate_expect_value(psi_e, a_rotated, lindbladian)

    distance = tf.abs(alpha0 - alpha1)
    iq_infid = tf.exp(-distance / d_max)
    return tf.abs(iq_infid)


@fid_reg_deco
def readout_and_clear(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]
    # Nr = params["Number_op"]
    clear_target_g = params["clear_target_ground"]
    clear_target_e = params["clear_target_excited"]
    readout_cost = params["readout_cost"]  # Penalty for bad readout
    clear_cost = params["clear_cost"]  # Penalty for bad clear

    psis_g = states[0]
    psis_e = states[1]

    alphas_g = calculate_expect_value(psis_g, a_rotated, lindbladian)
    alphas_e = calculate_expect_value(psis_e, a_rotated, lindbladian)

    distances = tf.abs(alphas_g - alphas_e)
    max_distance = tf.reduce_max(distances)
    readout_infid = tf.exp(-max_distance / d_max)

    # Ng_final = tf.abs(calculate_expect_value(psis_g[-1], Nr, lindbladian))
    # Ne_final = tf.abs(calculate_expect_value(psis_e[-1], Nr, lindbladian))

    overlap_g = calculate_state_overlap(psis_g[-1], clear_target_g)
    overlap_e = calculate_state_overlap(psis_e[-1], clear_target_e)

    clear_infid = 1 - (overlap_e + overlap_g) / 2

    # return readout_infid + Ng_final + Ne_final
    return readout_cost * readout_infid + clear_cost * clear_infid


@fid_reg_deco
def readout_and_clear_prod(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]
    clear_target_g = params["clear_target_ground"]
    clear_target_e = params["clear_target_excited"]
    clear_weight_g = params["clear_weight_ground"]
    clear_weight_e = params["clear_weight_excited"]
    readout_cost = params["readout_cost"]
    clear_cost = params["readout_cost"]

    psis_g = states[0]
    psis_e = states[1]

    alphas_g = calculate_expect_value(psis_g, a_rotated, lindbladian)
    alphas_e = calculate_expect_value(psis_e, a_rotated, lindbladian)

    distances = tf.abs(alphas_g - alphas_e)
    max_distance = tf.reduce_max(distances)
    readout_fid = (
        readout_cost * (1 - tf.exp(-max_distance / d_max)) / (readout_cost + clear_cost)
    )

    overlap_g = (
        clear_weight_g
        * calculate_state_overlap(psis_g[-1], clear_target_g)
        / (clear_weight_g + clear_weight_e)
    )
    overlap_e = (
        clear_weight_e
        * calculate_state_overlap(psis_e[-1], clear_target_e)
        / (clear_weight_g + clear_weight_e)
    )

    clear_fid = clear_cost * (overlap_e + overlap_g) / (readout_cost + clear_cost)

    return 1 - (readout_fid * clear_fid)


@fid_reg_deco
def swap_and_readout_ode(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]

    swap_position = params["swap_pos"]  # -1 for final state
    swap_cost = params["swap_cost"]
    swap_target_e = params["swap_target_state_excited"]
    swap_target_g = params["swap_target_state_ground"]

    swap_cost = tf.constant(swap_cost, dtype=tf.complex128)
    infid = tf.convert_to_tensor(0.0, dtype=tf.complex128)

    psi_g = states[0][-1]
    psi_e = states[1][-1]

    alpha0 = calculate_expect_value(psi_g, a_rotated, lindbladian)
    alpha1 = calculate_expect_value(psi_e, a_rotated, lindbladian)

    distance = tf.abs(alpha0 - alpha1)
    iq_infid = tf.exp(-distance / d_max)

    overlap_g = calculate_state_overlap(states[0][swap_position], swap_target_g)
    overlap_e = calculate_state_overlap(states[1][swap_position], swap_target_e)

    swap_infid = 1 - (overlap_e + overlap_g) / 2
    infid = swap_infid + iq_infid

    return tf.abs(infid)


@fid_reg_deco
def swap_and_readout_prod(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]

    swap_position = params["swap_pos"]  # -1 for final state
    swap_cost = params["swap_cost"]
    swap_target_e = params["swap_target_state_excited"]
    swap_target_g = params["swap_target_state_ground"]

    swap_cost = tf.constant(swap_cost, dtype=tf.complex128)
    infid = tf.convert_to_tensor(0.0, dtype=tf.complex128)

    psi_g = states[0][-1]
    psi_e = states[1][-1]

    alpha0 = calculate_expect_value(psi_g, a_rotated, lindbladian)
    alpha1 = calculate_expect_value(psi_e, a_rotated, lindbladian)

    distance = tf.abs(alpha0 - alpha1)
    iq_fid = 1 - tf.exp(-distance / d_max)

    overlap_g = calculate_state_overlap(states[0][swap_position], swap_target_g)
    overlap_e = calculate_state_overlap(states[1][swap_position], swap_target_e)

    swap_fid = (overlap_e + overlap_g) / 2
    infid = 1 - (swap_fid * iq_fid)

    return tf.abs(infid)


@fid_reg_deco
def readoutswap_trace_prod(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]

    swap_position = params["swap_pos"]  # -1 for final state
    swap_cost = params["swap_cost"]
    # swap_target_e = params["swap_target_state_excited"]
    # swap_target_g = params["swap_target_state_ground"]

    swap_cost = tf.constant(swap_cost, dtype=tf.complex128)
    infid = tf.convert_to_tensor(0.0, dtype=tf.complex128)

    psi_g = states[0][-1]
    psi_e = states[1][-1]

    alpha0 = calculate_expect_value(psi_g, a_rotated, lindbladian)
    alpha1 = calculate_expect_value(psi_e, a_rotated, lindbladian)

    distance = tf.abs(alpha0 - alpha1)
    iq_fid = 1 - tf.exp(-distance / d_max)

    ground_pop_g = partial_trace_two_systems(
        states[0][swap_position],
        tf.constant(dims, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )[0, 0]
    ground_pop_e = partial_trace_two_systems(
        states[1][swap_position],
        tf.constant(dims, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )[0, 0]

    ground_pop_g = tf.abs(ground_pop_g)
    ground_pop_e = tf.abs(ground_pop_e)

    swap_fid = (ground_pop_g + ground_pop_e) / 2
    infid = 1 - (swap_fid * iq_fid)

    return tf.abs(infid)


@fid_reg_deco
def qubit_reset(states: tf.Tensor, index, dims, params, n_eval=-1):
    psi_target = params["target"]

    overlaps = calculate_state_overlap_batch(states, psi_target)
    max_overlap = tf.reduce_max(overlaps)

    infid = 1 - max_overlap
    return tf.abs(infid)


@fid_reg_deco
def readout_clear_swap_prod(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]
    swap_target_g = params["swap_target_ground"]
    swap_target_e = params["swap_target_excited"]

    psis_g = states[0]
    psis_e = states[1]

    alphas_g = calculate_expect_value(psis_g, a_rotated, lindbladian)
    alphas_e = calculate_expect_value(psis_e, a_rotated, lindbladian)

    distances = tf.abs(alphas_g - alphas_e)
    max_distance = tf.reduce_max(distances)
    readout_fid = 1 - tf.exp(-max_distance / d_max)

    overlap_g = calculate_state_overlap(psis_g[-1], swap_target_g)
    overlap_e = calculate_state_overlap(psis_e[-1], swap_target_e)

    swap_fid = (overlap_e + overlap_g) / 2

    return 1 - (readout_fid * swap_fid)


@fid_reg_deco
def reset_ptrace(states: tf.Tensor, index, dims, params=None, n_eval=-1):
    """
    Trace out the resonator and calculate the ground state occupation of the qubit.
    """

    rho_qubit = partial_trace_two_systems(
        states[-1],
        tf.constant(dims, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )

    ground_state_pop = tf.abs(rho_qubit[0, 0])

    return 1 - ground_state_pop


@fid_reg_deco
def reset_schrodinger(states: tf.Tensor, index, dims, params=None, n_eval=-1):
    """
    Trace out the resonator and calculate the ground state occupation of the qubit.
    """
    rho_final = tf_state_to_dm(states[-1])

    rho_qubit = partial_trace_two_systems(
        rho_final,
        tf.constant(dims, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )

    ground_state_pop = tf.abs(rho_qubit[0, 0])

    return 1 - ground_state_pop


@fid_reg_deco
def readout_and_clear_ground(states: tf.Tensor, index, dims, params, n_eval=-1):
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]

    psis_g = states[0]
    psis_e = states[1]

    alphas_g = calculate_expect_value(psis_g, a_rotated, lindbladian)
    alphas_e = calculate_expect_value(psis_e, a_rotated, lindbladian)

    distances = tf.abs(alphas_g - alphas_e)
    max_distance = tf.reduce_max(distances)
    readout_fid = 1 - tf.exp(-max_distance / d_max)

    clear_fid = 1 - tf.abs(alphas_g[-1]) / tf.reduce_max(tf.abs(alphas_g))

    return 1 - (readout_fid * clear_fid)


@fid_reg_deco
def readout_and_clear_ground_2(states: tf.Tensor, index, dims, params, n_eval=-1):
    """
    Optimize for greater IQ plane distance between the ground and excited state
    at some time and then at the end clear the resonator population only if the
    qubit started in the ground state. Here I am using the overlap with the
    state with empty resonator as cost function as the previous method may
    depend on the rotating frame of reference.

    Args:
        states (tf.Tensor): [psis_g, psis_e]
        index (_type_): -
        dims (_type_): subsystem dimensions
        params (_type_): {a_rotated, cutoff_distance, lindbladian, clear_target_ground}
        n_eval (int, optional): _description_. Defaults to -1.

    """
    print("Calculating fidelity")
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]
    clear_target_g = params["clear_target_ground"]

    psis_g = states[0]
    psis_e = states[1]

    alphas_g = calculate_expect_value(psis_g, a_rotated, lindbladian)
    alphas_e = calculate_expect_value(psis_e, a_rotated, lindbladian)

    distances = tf.abs(alphas_g - alphas_e)
    max_distance = tf.reduce_max(distances)
    readout_fid = 1 - tf.exp(-max_distance / d_max)

    clear_fid = calculate_state_overlap(psis_g[-1], clear_target_g)

    return 1 - (readout_fid * clear_fid)


@fid_reg_deco
def remove_leakage(states: tf.Tensor, index, dims, params=None, n_eval=-1):
    """
    Trace out the resonator and calculate the ground state occupation of the qubit.
    """

    rho_qubit = partial_trace_two_systems(
        states[-1],
        tf.constant(dims, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )

    leakage_pop = 1 - tf.abs(rho_qubit[0, 0] + rho_qubit[1, 1])

    return leakage_pop


@fid_reg_deco
def remove_leakage_multi_state(states: tf.Tensor, index, dims, params=None, n_eval=-1):
    """
    Trace out the resonator and calculate the ground state occupation of the qubit.
    """

    final_states = states[-1]
    infids = tf.TensorArray(
        tf.float64, size=final_states.shape[0], dynamic_size=False, infer_shape=False
    )

    for i in tf.range(final_states.shape[0]):
        rho_qubit = partial_trace_two_systems(
            final_states[i],
            tf.constant(dims, dtype=tf.int32),
            tf.constant(1, dtype=tf.int32),
        )

        leakage_pop = 1 - tf.abs(rho_qubit[0, 0] + rho_qubit[1, 1])
        infids = infids.write(i, leakage_pop)

    infids = infids.stack()

    return tf.reduce_mean(infids)

@fid_reg_deco
def reset_ptrace_multi_state(states: tf.Tensor, index, dims, params=None, n_eval=-1):
    """
    Trace out the resonator and calculate the ground state occupation of the qubit.
    """

    final_states = states[-1]
    infids = tf.TensorArray(
        tf.float64, size=final_states.shape[0], dynamic_size=False, infer_shape=False
    )

    for i in tf.range(final_states.shape[0]):
        rho_qubit = partial_trace_two_systems(
            final_states[i],
            tf.constant(dims, dtype=tf.int32),
            tf.constant(1, dtype=tf.int32),
        )

        leakage_pop = 1 - tf.abs(rho_qubit[0, 0])
        infids = infids.write(i, leakage_pop)

    infids = infids.stack()

    return tf.reduce_mean(infids)


@fid_reg_deco
def multi_state_infidelity(states: tf.Tensor, index, dims, params, n_eval=-1):
    """
    Define a cost function that calculates the inifidelity for individual states
    when starting with multiple initial states.

    Here the states have the shape [times, number of initial states, dim, dim]

    The params dictionary must contain:
        fid_function: A fidelity function that computes the infidelity for each init state
        weights: a weighting function that specifies the weights of each of the init state during optimization
        params_list: List of params dictionary to pe passed onto the fid_function incase it requires extra params

    Args:
        states (tf.Tensor): Simulated states of the above mentioned shape
        index (_type_): -
        dims (_type_): model.dims
        params (dict): As mentioned above
        n_eval (int, optional): _description_. Defaults to -1.
    """

    fid_function = params["fid_function"]
    weights = params["weights"]
    weights = weights / tf.reduce_sum(weights)  # Normalizing the weights
    params_list = params["params_list"]

    num_init_states = states.shape[1]
    infids = tf.TensorArray(
        tf.float64, size=num_init_states, dynamic_size=False, infer_shape=False
    )
    for i in range(num_init_states):
        infid = fid_function(
            states=states[:, i, ...],
            index=1,
            dims=dims,
            params=params_list[i],
            n_eval=n_eval,
        )

        infids = infids.write(i, weights[i] * infid)

    infids = infids.stack()
    return tf.reduce_sum(infids)


@fid_reg_deco
def state_transfer_ptrace(states: tf.Tensor, index, dims, params, n_eval=-1):
    target_index = params["target_index"]

    rho_qubit = partial_trace_two_systems(
        states[-1],
        tf.constant(dims, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
    )

    target_pop = tf.abs(rho_qubit[target_index, target_index])
    return 1 - target_pop


@fid_reg_deco
def readout_and_clear_ground_simultaneous(
    states: tf.Tensor, index, dims, params, n_eval=-1
):
    """
    Here I do readout optimization by using a multi-init
    and not running the whole simulation seperately for the ground and excited state.
    For this optimization keep `optimalcontrol.readout` as `False`.

    Args:
        states (tf.Tensor): [psis_g, psis_e]
        index (_type_): -
        dims (_type_): subsystem dimensions
        params (_type_): {a_rotated, cutoff_distance, lindbladian, clear_target_ground}
        n_eval (int, optional): _description_. Defaults to -1.

    """
    a_rotated = params["a_rotated"]
    d_max = params["cutoff_distance"]
    lindbladian = params["lindbladian"]
    clear_target_g = params["clear_target_ground"]
    clear_target_e = params["clear_target_excited"]

    psis_g = states[:, 0, ...]
    psis_e = states[:, 1, ...]

    alphas_g = calculate_expect_value(psis_g, a_rotated, lindbladian)
    alphas_e = calculate_expect_value(psis_e, a_rotated, lindbladian)

    distances = tf.abs(alphas_g - alphas_e)
    max_distance = tf.reduce_max(distances)
    readout_fid = 1 - tf.exp(-max_distance / d_max)

    clear_fid_g = calculate_state_overlap(psis_g[-1], clear_target_g)
    clear_fid_e = calculate_state_overlap(psis_e[-1], clear_target_e)

    clear_fid = (5 * clear_fid_g + clear_fid_e) / 6  # fixing these values for now

    return 1 - (readout_fid * clear_fid)


def compute_SNR(states, a_op, dt, eta, kappa):
    """
    Compute the signal to noise ratio (SNR) for the readout process
    """
    psis_g = states[:, 0, ...]
    psis_e = states[:, 1, ...]

    alphas_g = calculate_expect_value(psis_g, a_op, lindbladian=True)
    alphas_e = calculate_expect_value(psis_e, a_op, lindbladian=True)

    diff = tf.abs(alphas_e - alphas_g) ** 2
    integral = tf.reduce_sum(diff) * dt

    SNR = tf.sqrt(2 * eta * kappa * integral)

    return SNR


@fid_reg_deco
def readout_and_clear_SNR(states: tf.Tensor, index, dims, params, n_eval=-1):
    """
    Here I do readout optimization by using a multi-init
    and not running the whole simulation seperately for the ground and excited state.
    For this optimization keep `optimalcontrol.readout` as `False`.

    Args:
        states (tf.Tensor): [psis_g, psis_e]
        index (_type_): -
        dims (_type_): subsystem dimensions
        params (_type_): {a_op, clear_target_ground, clear_target_excited, sim_res, eta, kappa}
        n_eval (int, optional): _description_. Defaults to -1.

    """
    a_op = params["a_op"]
    clear_target_g = params["clear_target_ground"]
    clear_target_e = params["clear_target_excited"]
    dt = 1 / params["sim_res"]
    eta = params["eta"]
    kappa = params["kappa"]

    SNR = compute_SNR(states, a_op, dt, eta, kappa)
    readout_fid = 1 - tf.exp(-SNR)

    psis_g = states[:, 0, ...]
    psis_e = states[:, 1, ...]

    clear_fid_g = calculate_state_overlap(psis_g[-1], clear_target_g)
    clear_fid_e = calculate_state_overlap(psis_e[-1], clear_target_e)

    clear_fid = (5 * clear_fid_g + clear_fid_e) / 6  # fixing these values for now

    return 1 - (readout_fid * clear_fid)
