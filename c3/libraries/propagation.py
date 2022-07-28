"A library for propagators and closely related functions"
from posixpath import split
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict
from c3.model import Model
from c3.generator.generator import Generator
from c3.signal.gates import Instruction
from scipy import interpolate
from c3.utils.tf_utils import (
    tf_kron,
    tf_matmul_left,
    tf_matmul_n,
    tf_spre,
    tf_spost,
)
from c3.libraries.constants import kb, hbar

unitary_provider = dict()
state_provider = dict()


def step_vonNeumann_psi(psi, h, dt):
    return -1j * dt * tf.linalg.matmul(h, psi)


def unitary_deco(func):
    """
    Decorator for making registry of functions
    """
    unitary_provider[str(func.__name__)] = func
    return func


def state_deco(func):
    """
    Decorator for making registry of functions
    """
    state_provider[str(func.__name__)] = func
    return func


@unitary_deco
def gen_dus_rk4(h, dt, dim=None):
    dUs = []
    dU = []
    if dim is None:
        tot_dim = tf.shape(h)
        dim = tot_dim[1]

    for jj in range(0, len(h) - 2, 2):
        dU = gen_du_rk4(h[jj : jj + 3], dt, dim)
        dUs.append(dU)
    return dUs


def gen_du_rk4(h, dt, dim):
    temp = []
    for ii in range(dim):
        psi = tf.one_hot(ii, dim, dtype=tf.complex128)
        psi = rk4_step(h, psi, dt)
        temp.append(psi)
    dU = tf.stack(temp)
    return dU


def rk4_step(h, psi, dt):
    k1 = step_vonNeumann_psi(psi, h[0], dt)
    k2 = step_vonNeumann_psi(psi + k1 / 2.0, h[1], dt)
    k3 = step_vonNeumann_psi(psi + k2 / 2.0, h[1], dt)
    k4 = step_vonNeumann_psi(psi + k3, h[2], dt)
    psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return psi


def get_hs_of_t_ts(
    model: Model, gen: Generator, instr: Instruction, prop_res=1
) -> Dict:
    if model.controllability:
        hs_of_ts = _get_hs_of_t_ts_controlled(model, gen, instr, prop_res)
    else:
        hs_of_ts = _get_hs_of_t_ts(model, gen, instr, prop_res)
    return hs_of_ts


def _get_hs_of_t_ts_controlled(
    model: Model, gen: Generator, instr: Instruction, prop_res=1
) -> Dict:
    """
    Return a Dict containing:

    - a list of

      H(t) = H_0 + sum_k c_k H_k.

    - time slices ts

    - timestep dt

    Parameters
    ----------
    prop_res : tf.float
        resolution required by the propagation method
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    ts : float
        Length of one time slice.
    """
    Hs = []
    ts = []
    gen.resolution = prop_res * gen.resolution
    signal = gen.generate_signals(instr)
    h0, hctrls = model.get_Hamiltonians()
    signals = []
    hks = []
    for key in signal:
        signals.append(signal[key]["values"])
        ts = signal[key]["ts"]
        hks.append(hctrls[key])
    cflds = tf.cast(signals, tf.complex128)
    hks = tf.cast(hks, tf.complex128)
    for ii in range(cflds[0].shape[0]):
        cf_t = []
        for fields in cflds:
            cf_t.append(tf.cast(fields[ii], tf.complex128))
        Hs.append(sum_h0_hks(h0, hks, cf_t))

    dt = tf.constant(ts[1 * prop_res].numpy() - ts[0].numpy(), dtype=tf.complex128)
    return {"Hs": Hs, "ts": ts[::prop_res], "dt": dt}


def _get_hs_of_t_ts(
    model: Model, gen: Generator, instr: Instruction, prop_res=1
) -> Dict:
    """
    Return a Dict containing:

    - a list of

      H(t) = H_0 + sum_k c_k H_k.

    - time slices ts

    - timestep dt

    Parameters
    ----------
    prop_res : tf.float
        resolution required by the propagation method
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    ts : float
        Length of one time slice.
    """
    Hs = []
    ts = []
    gen.resolution = prop_res * gen.resolution
    signal = gen.generate_signals(instr)
    Hs = model.get_Hamiltonian(signal)
    ts_list = [sig["ts"][1:] for sig in signal.values()]
    ts = tf.constant(tf.math.reduce_mean(ts_list, axis=0))
    if not np.all(tf.math.reduce_variance(ts_list, axis=0) < 1e-5 * (ts[1] - ts[0])):
        raise Exception("C3Error:Something with the times happend.")
    if not np.all(tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])):  # type: ignore
        raise Exception("C3Error:Something with the times happend.")

    dt = tf.constant(ts[1 * prop_res].numpy() - ts[0].numpy(), dtype=tf.complex128)
    return {"Hs": Hs, "ts": ts[::prop_res], "dt": dt}


def sum_h0_hks(h0, hks, cf_t):
    """
    Compute and Return

     H(t) = H_0 + sum_k c_k H_k.
    """
    h_of_t = h0
    ii = 0
    while ii < len(hks):
        h_of_t += cf_t[ii] * hks[ii]
        ii += 1
    return h_of_t


@unitary_deco
def rk4(model: Model, gen: Generator, instr: Instruction, init_state=None) -> Dict:
    prop_res = 2
    dim = model.tot_dim
    Hs = []
    ts = []
    dUs = []
    dict_vals = get_hs_of_t_ts(model, gen, instr, prop_res)
    Hs = dict_vals["Hs"]
    ts = dict_vals["ts"]
    dt = dict_vals["dt"]

    dUs = gen_dus_rk4(Hs, dt, dim)

    U = gen_u_rk4(Hs, dt, dim)

    if model.max_excitations:
        U = model.blowup_excitations(U)
        dUs = tf.vectorized_map(model.blowup_excitations, dUs)

    return {"U": U, "dUs": dUs, "ts": ts}


def gen_u_rk4(h, dt, dim):
    U = []
    for ii in range(dim):
        psi = tf.one_hot(ii, dim, dtype=tf.complex128)

        for jj in range(0, len(h) - 2, 2):
            psi = rk4_step(h[jj : jj + 3], psi, dt)
        U.append(psi)
    U = tf.stack(U)
    return tf.transpose(U)


@unitary_deco
def pwc(
    model: Model,
    gen: Generator,
    instr: Instruction,
    folding_stack: list,
    batch_size=None,
) -> Dict:
    """
    Solve the equation of motion (Lindblad or Schrรถdinger) for a given control
    signal and Hamiltonians.

    Parameters
    ----------
    signal: dict
        Waveform of the control signal per drive line.
    gate: str
        Identifier for one of the gates.

    Returns
    -------
    unitary
        Matrix representation of the gate.
    """
    signal = gen.generate_signals(instr)
    # Why do I get 0.0 if I print gen.resolution here?! FR
    ts = []
    if model.controllability:
        h0, hctrls = model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            ts = signal[key]["ts"]
            hks.append(hctrls[key])
        signals = tf.cast(signals, tf.complex128)
        hks = tf.cast(hks, tf.complex128)
    else:
        h0 = model.get_Hamiltonian(signal)
        ts_list = [sig["ts"][1:] for sig in signal.values()]
        ts = tf.constant(tf.math.reduce_mean(ts_list, axis=0))
        hks = None
        signals = None
        if not np.all(
            tf.math.reduce_variance(ts_list, axis=0) < 1e-5 * (ts[1] - ts[0])
        ):
            raise Exception("C3Error:Something with the times happend.")
        if not np.all(
            tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])  # type: ignore
        ):
            raise Exception("C3Error:Something with the times happend.")

    dt = ts[1] - ts[0]

    if batch_size is None:
        batch_size = tf.constant(len(h0), tf.int32)
    else:
        batch_size = tf.constant(batch_size, tf.int32)

    if model.lindbladian:
        col_ops = model.get_Lindbladians()
        if model.max_excitations:
            cutter = model.ex_cutter
            col_ops = [cutter @ col_op @ cutter.T for col_op in col_ops]
        dUs = tf_batch_propagate(
            h0,
            hks,
            signals,
            dt,
            batch_size=batch_size,
            col_ops=col_ops,
            lindbladian=True,
        )
    else:
        dUs = tf_batch_propagate(h0, hks, signals, dt, batch_size=batch_size)

    # U = tf_matmul_left(tf.cast(dUs, tf.complex128))
    U = tf_matmul_n(dUs, folding_stack)

    if model.max_excitations:
        U = model.blowup_excitations(tf_matmul_left(tf.cast(dUs, tf.complex128)))
        dUs = tf.vectorized_map(model.blowup_excitations, dUs)

    return {"U": U, "dUs": dUs, "ts": ts}

####################
# HELPER FUNCTIONS #
####################


def tf_dU_of_t(h0, hks, cflds_t, dt):
    """
    Compute H(t) = H_0 + sum_k c_k H_k and matrix exponential exp(-i H(t) dt).

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    dt : float
        Length of one time slice.

    Returns
    -------
    tf.tensor
        dU = exp(-i H(t) dt)

    """
    h = h0
    ii = 0
    while ii < len(hks):
        h += cflds_t[ii] * hks[ii]
        ii += 1
    # terms = int(1e12 * dt) + 2
    # dU = tf_expm(-1j * h * dt, terms)
    # TODO Make an option for the exponentation method
    dU = tf.linalg.expm(-1j * h * dt)
    return dU


def tf_dU_of_t_lind(h0, hks, col_ops, cflds_t, dt):
    """
    Compute the Lindbladian and it's matrix exponential exp(L(t) dt).

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    col_ops : list of tf.tensor
        List of collapse operators.
    cflds_t : array of tf.float
        Vector of control field values at time t.
    dt : float
        Length of one time slice.

    Returns
    -------
    tf.tensor
        dU = exp(L(t) dt)

    """
    h = h0
    for ii in range(len(hks)):
        h += cflds_t[ii] * hks[ii]
    lind_op = -1j * (tf_spre(h) - tf_spost(h))
    for col_op in col_ops:
        super_clp = tf.matmul(tf_spre(col_op), tf_spost(tf.linalg.adjoint(col_op)))
        anticomm_L_clp = 0.5 * tf.matmul(
            tf_spre(tf.linalg.adjoint(col_op)), tf_spre(col_op)
        )
        anticomm_R_clp = 0.5 * tf.matmul(
            tf_spost(col_op), tf_spost(tf.linalg.adjoint(col_op))
        )
        lind_op = lind_op + super_clp - anticomm_L_clp - anticomm_R_clp
    # terms = int(1e12 * dt) # Eyeball number of terms in expm
    #     print('terms in exponential: ', terms)
    # dU = tf_expm(lind_op * dt, terms)
    # Built-in tensorflow exponential below
    dU = tf.linalg.expm(lind_op * dt)
    return dU


def tf_propagation_vectorized(h0, hks, cflds_t, dt):
    dt = tf.cast(dt, dtype=tf.complex128)
    if hks is not None and cflds_t is not None:
        cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
        hks = tf.cast(hks, dtype=tf.complex128)
        cflds = tf.expand_dims(tf.expand_dims(cflds_t, 2), 3)
        hks = tf.expand_dims(hks, 1)
        if len(h0.shape) < 3:
            h0 = tf.expand_dims(h0, 0)
        prod = cflds * hks
        h = h0 + tf.reduce_sum(prod, axis=0)
    else:
        h = tf.cast(h0, tf.complex128)
    dh = -1.0j * h * dt
    return tf.linalg.expm(dh)


def pwc_trott_drift(h0, hks, cflds_t, dt):
    dt = tf.cast(dt, dtype=tf.complex128)
    cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
    hks = tf.cast(hks, dtype=tf.complex128)
    e, v = tf.linalg.eigh(h0)
    ort = tf.cast(v, dtype=tf.complex128)
    dE = tf.math.exp(-1.0j * tf.math.real(e) * dt)
    dU0 = ort @ tf.linalg.diag(dE) @ ort.T
    prod = cflds_t * hks
    ht = tf.reduce_sum(prod, axis=0)
    comm = h0 @ ht - ht @ h0
    dh = -1.0j * ht * dt
    dcomm = -comm * dt**2 / 2.0
    dUs = dU0 @ tf.linalg.expm(dh) @ (tf.identity(dU0) - dcomm)
    return dUs


def tf_batch_propagate(
    hamiltonian, hks, signals, dt, batch_size, col_ops=None, lindbladian=False
):
    """
    Propagate signal in batches
    Parameters
    ----------
    hamiltonian: tf.tensor
        Drift Hamiltonian
    hks: Union[tf.tensor, List[tf.tensor]]
        List of control hamiltonians
    signals: Union[tf.tensor, List[tf.tensor]]
        List of control signals, one per control hamiltonian
    dt: float
        Length of one time slice
    batch_size: int
        Number of elements in one batch

    Returns
    -------

    """
    if signals is not None:
        batches = int(tf.math.ceil(signals.shape[1] / batch_size))
        batch_array = tf.TensorArray(
            signals.dtype, size=batches, dynamic_size=False, infer_shape=False
        )
        for i in range(batches):
            batch_array = batch_array.write(
                i, signals[:, i * batch_size : i * batch_size + batch_size]
            )
    else:
        batches = int(tf.math.ceil(hamiltonian.shape[0] / batch_size))
        batch_array = tf.TensorArray(
            hamiltonian.dtype, size=batches, dynamic_size=False, infer_shape=False
        )
        for i in range(batches):
            batch_array = batch_array.write(
                i, hamiltonian[i * batch_size : i * batch_size + batch_size]
            )

    dUs_array = tf.TensorArray(tf.complex128, size=batches, infer_shape=False)
    for i in range(batches):
        x = batch_array.read(i)
        if signals is not None:
            if lindbladian:
                result = tf_propagation_lind(hamiltonian, hks, col_ops, x, dt)
            else:
                result = tf_propagation_vectorized(hamiltonian, hks, x, dt)
        else:
            if lindbladian:
                # TODO - Check if it works
                result = tf_propagation_lind(x, None, None, None, dt)
            else:
                result = tf_propagation_vectorized(x, None, None, dt)
        dUs_array = dUs_array.write(i, result)
    return dUs_array.concat()


@unitary_deco
def tf_propagation(h0, hks, cflds, dt):
    """
    Calculate the unitary time evolution of a system controlled by time-dependent
    fields.

    Parameters
    ----------
    h0 : tf.tensor
        Drift Hamiltonian.
    hks : list of tf.tensor
        List of control Hamiltonians.
    cflds : list
        List of control fields, one per control Hamiltonian.
    dt : float
        Length of one time slice.

    Returns
    -------
    list
        List of incremental propagators dU.

    """
    dUs = []

    for ii in range(cflds[0].shape[0]):
        cf_t = []
        for fields in cflds:
            cf_t.append(tf.cast(fields[ii], tf.complex128))
        dUs.append(tf_dU_of_t(h0, hks, cf_t, dt))
    return dUs


def tf_propagation_lind(h0, hks, col_ops, cflds_t, dt, history=False):
    col_ops = tf.cast(col_ops, dtype=tf.complex128)
    dt = tf.cast(dt, dtype=tf.complex128)
    if hks is not None and cflds_t is not None:
        cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
        hks = tf.cast(hks, dtype=tf.complex128)
        cflds = tf.expand_dims(tf.expand_dims(cflds_t, 2), 3)
        hks = tf.expand_dims(hks, 1)
        h0 = tf.expand_dims(h0, 0)
        prod = cflds * hks
        h = h0 + tf.reduce_sum(prod, axis=0)
    else:
        h = h0

    h_id = tf.eye(tf.shape(h)[-1], batch_shape=[tf.shape(h)[0]], dtype=tf.complex128)
    l_s = tf_kron(h, h_id)
    r_s = tf_kron(h_id, tf.linalg.matrix_transpose(h))
    lind_op = -1j * (l_s - r_s)

    col_ops_id = tf.eye(
        col_ops.shape[-1], batch_shape=[col_ops.shape[0]], dtype=tf.complex128
    )
    l_col_ops = tf_kron(col_ops, col_ops_id)
    r_col_ops = tf_kron(col_ops_id, tf.linalg.matrix_transpose(col_ops))

    super_clp = tf.matmul(l_col_ops, r_col_ops, adjoint_b=True)
    anticom_L_clp = 0.5 * tf.matmul(l_col_ops, l_col_ops, adjoint_a=True)
    anticom_R_clp = 0.5 * tf.matmul(r_col_ops, r_col_ops, adjoint_b=True)
    clp = tf.expand_dims(
        tf.reduce_sum(super_clp - anticom_L_clp - anticom_R_clp, axis=0), 0
    )
    lind_op += clp

    dU = tf.linalg.expm(lind_op * dt)
    return dU


def evaluate_sequences(propagators: Dict, sequences: list):
    """
    Compute the total propagator of a sequence of gates.

    Parameters
    ----------
    propagators : dict
        Dictionary of unitary representation of gates.

    sequences : list
        List of keys from propagators specifying a gate sequence.
        The sequence is multiplied from the left, i.e.
            sequence = [U0, U1, U2, ...]
        is applied as
            ... U2 * U1 * U0

    Returns
    -------
    tf.tensor
        Propagator of the sequence.

    """
    gates = propagators
    # get dims to deal with the case where a sequence is empty
    dim = list(gates.values())[0].shape[0]
    dtype = list(gates.values())[0].dtype
    # TODO deal with the case where you only evaluate one sequence
    U = []
    for sequence in sequences:
        if len(sequence) == 0:
            U.append(tf.linalg.eye(dim, dtype=dtype))
        else:
            Us = []
            for gate in sequence:
                Us.append(gates[gate])

            Us = tf.cast(Us, tf.complex128)
            U.append(tf_matmul_left(Us))
            # ### WARNING WARNING ^^ look there, it says left WARNING
    return U


def tf_expm(A, terms):
    """
    Matrix exponential by the series method.

    Parameters
    ----------
    A : tf.tensor
        Matrix to be exponentiated.
    terms : int
        Number of terms in the series.

    Returns
    -------
    tf.tensor
        expm(A)

    """
    r = tf.eye(int(A.shape[-1]), batch_shape=A.shape[:-2], dtype=A.dtype)
    A_powers = A
    r += A

    for ii in range(2, terms):
        A_powers = tf.matmul(A_powers, A) / tf.cast(ii, tf.complex128)
        ii += 1
        r += A_powers
    return r


def tf_expm_dynamic(A, acc=1e-5):
    """
    Matrix exponential by the series method with specified accuracy.

    Parameters
    ----------
    A : tf.tensor
        Matrix to be exponentiated.
    acc : float
        Accuracy. Stop when the maximum matrix entry reaches

    Returns
    -------
    tf.tensor
        expm(A)

    """
    r = tf.eye(int(A.shape[0]), dtype=A.dtype)
    A_powers = A
    r += A

    ii = tf.constant(2, dtype=tf.complex128)
    while tf.reduce_max(tf.abs(A_powers)) > acc:
        A_powers = tf.matmul(A_powers, A) / ii
        ii += 1
        r += A_powers
    return r


@state_deco
def lindblad_rk4(
    model: Model,
    gen: Generator,
    instr: Instruction,
    collapse_ops: tf.Tensor,
    init_state=None,
    solver="rk4",
) -> Dict:

    if solver == "rk4":
        interpolate_res = 2
    elif solver == "rk38":
        interpolate_res = 3
    elif solver == "rk5":
        interpolate_res = -5 # Fixing this a random number for now
    elif solver == "Tsit5":
        interpolate_res = -6 # Fixing this a random number for now

    Hs_dict = Hs_of_t(model, gen, instr, interpolate_res=interpolate_res)
    Hs = Hs_dict["Hs"]
    ts = Hs_dict["ts"]
    dt = Hs_dict["dt"]
    rhos = propagate_lind(Hs, collapse_ops, init_state, ts, dt, solver=solver)

    return {"states": rhos, "ts": ts}


def Hs_of_t(model, gen, instr, interpolate_res=2, L_dag_L=None):
    if model.controllability:
        Hs = get_Hs_of_t_cflds(model, gen, instr, interpolate_res, L_dag_L)
    else:
        Hs = get_Hs_of_t_no_cflds(model, gen, instr, interpolate_res)
    return Hs


def get_Hs_of_t_cflds(model, gen, instr, interpolate_res, L_dag_L=None):
    ts = []
    signal = gen.generate_signals(instr)
    h0, hctrls = model.get_Hamiltonians()
    signals = []
    hks = []
    for key in signal:
        signals.append(signal[key]["values"])
        ts = signal[key]["ts"]
        hks.append(hctrls[key])
    signals_interp = []
    for sig in signals:
        sig_new = interpolateSignal(ts, sig, interpolate_res)
        signals_interp.append(sig_new)

    cflds = tf.cast(signals_interp, tf.complex128)
    hks = tf.cast(hks, tf.complex128)
    if L_dag_L is not None:
        Hs = calculate_sum_Hs(h0, hks, cflds, L_dag_L)
    else:
        Hs = calculate_sum_Hs(h0, hks, cflds)
    ts = tf.cast(ts, dtype=tf.complex128)
    dt = ts[1] - ts[0]

    if L_dag_L is not None:
        return {"Hs": Hs, "ts": ts, "dt": dt, "LdagL": L_dag_L} 
    else:
        return {"Hs": Hs, "ts": ts, "dt": dt}

def calculate_sum_Hs(h0, hks, cflds, L_dag_L=None):
    control_field = tf.reshape(
        tf.transpose(cflds), 
        (tf.shape(cflds)[1], tf.shape(cflds)[0], 1, 1)
    )
    hk = tf.multiply(control_field, hks)
    Hs = tf.reduce_sum(hk, axis=1)
    if L_dag_L is not None:
        return Hs + h0 -1j * 0.5 * tf.reduce_sum(tf.reduce_sum(L_dag_L, axis=0), axis=0)
    return Hs + h0

# TODO - change this function to include interpolation
# TODO - Also make this compatible with tf.function by removing .numpy()
def get_Hs_of_t_no_cflds(model, gen, instr, prop_res):
    Hs = []
    ts = []
    gen.resolution = prop_res * gen.resolution
    signal = gen.generate_signals(instr)
    Hs = model.get_Hamiltonian(signal)
    ts_list = [sig["ts"][1:] for sig in signal.values()]
    ts = tf.constant(tf.math.reduce_mean(ts_list, axis=0))
    if not np.all(tf.math.reduce_variance(ts_list, axis=0) < 1e-5 * (ts[1] - ts[0])):
        raise Exception("C3Error:Something with the times happend.")
    if not np.all(tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])):  # type: ignore
        raise Exception("C3Error:Something with the times happend.")

    dt = tf.constant(ts[1 * prop_res].numpy() - ts[0].numpy(), dtype=tf.complex128)
    return {"Hs": Hs, "ts": ts[::prop_res], "dt": dt}

def propagate_lind(Hs, col, rho, ts, dt, solver="rk4"):
    rho_list = tf.TensorArray(
                    tf.complex128, 
                    size=ts.shape[0], 
                    dynamic_size=False, 
                    infer_shape=False
    )
    rho_t = rho
    for index in tf.range(ts.shape[0]):
        if solver =="rk38":
            h = tf.slice(Hs, [3*index, 0, 0], [4, Hs.shape[1], Hs.shape[2]])
            rho_t = rk38_step_lind(lindblad_step, rho_t, h, dt, col=col)
        elif solver == "rk5":
            h = tf.slice(Hs, [6*index, 0, 0], [6, Hs.shape[1], Hs.shape[2]])
            rho_t = rk5_dopri_step_lind(lindblad_step, rho_t, h, dt, col=col)
        elif solver == "Tsit5":
            h = tf.slice(Hs, [6*index, 0, 0], [6, Hs.shape[1], Hs.shape[2]])
            rho_t = Tsit5_step_lind(lindblad_step, rho_t, h, dt, col=col)
        else:
            h = tf.slice(Hs, [2*index, 0, 0], [3, Hs.shape[1], Hs.shape[2]])
            rho_t = rk4_step_lind(lindblad_step, rho_t, h, dt, col=col)
        rho_list = rho_list.write(index, rho_t)
    return rho_list.stack()

# TODO - make a RK45 algorithm with interpolation
def rk4_step_lind(func, rho, h, dt, col=None):
    if col == None:
        k1 = func(rho, h[0], dt)
        k2 = func(rho + k1 / 2.0, h[1], dt)
        k3 = func(rho + k2 / 2.0, h[1], dt)
        k4 = func(rho + k3, h[2], dt)
    else:
        k1 = func(rho, h[0], col, dt)
        k2 = func(rho + k1 / 2.0, h[1], col, dt)
        k3 = func(rho + k2 / 2.0, h[1], col, dt)
        k4 = func(rho + k3, h[2], col, dt)
    rho_new = rho + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return rho_new

def rk38_step_lind(func, rho, h, dt, col=None):
    if col == None:
        k1 = func(rho, h[0], dt)
        k2 = func(rho + k1 / 3.0, h[1], dt)
        k3 = func(rho + (-k1 / 3.0) + k2, h[2], dt)
        k4 = func(rho + k1 - k2 + k3, h[3], dt)
    else:
        k1 = func(rho, h[0], col, dt)
        k2 = func(rho + k1 / 3.0, h[1], col, dt)
        k3 = func(rho + (-k1 / 3.0) + k2, h[2], col, dt)
        k4 = func(rho + k1 -k2 + k3, h[3], col, dt)
    rho_new = rho + (k1 + 3 * k2 + 3 * k3 + k4) / 8.0
    return rho_new

def rk5_dopri_step_lind(func, rho, h, dt, col=None):
    if col == None:
        k1 = func(rho, h[0], dt)
        k2 = func(rho + 1./5 *k1, h[1], dt)
        k3 = func(rho + 3./40*k1 + 9./40*k2, h[2], dt)
        k4 = func(rho + 44./45*k1 - 56./15*k2 + 32./9*k3, h[3], dt)
        k5 = func(rho + 19372./6561*k1 - 25360./2187*k2 + 64448./6561*k3 - 212./729*k4, h[4], dt)
        k6 = func(rho + 9017./3168*k1 - 355./33*k2 + 46732./5247*k3 + 49./176*k4 - 5103./18656*k5, h[5], dt)
        k7 = func(rho + 35./384*k1 + 500./1113*k3 + 125./192*k4 - 2187./6784*k5 + 11./84*k6, h[5], dt)
    else:
        k1 = func(rho, h[0], col, dt)
        k2 = func(rho + 1./5 *k1, h[1], col, dt)
        k3 = func(rho + 3./40*k1 + 9./40*k2, h[2], col, dt)
        k4 = func(rho + 44./45*k1 - 56./15*k2 + 32./9*k3, h[3], col, dt)
        k5 = func(rho + 19372./6561*k1 - 25360./2187*k2 + 64448./6561*k3 - 212./729*k4, h[4], col, dt)
        k6 = func(rho + 9017./3168*k1 - 355./33*k2 + 46732./5247*k3 + 49./176*k4 - 5103./18656*k5, h[5], col, dt)
        k7 = func(rho + 35./384*k1 + 500./1113*k3 + 125./192*k4 - 2187./6784*k5 + 11./84*k6, h[5], col, dt)
        
    rho_new = rho + 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 - 92097./339200*k5 + 187./2100*k6 + 1./40*k7
    return rho_new

def Tsit5_step_lind(func, rho, h, dt, col=None):
    if col == None:
        k1 = func(rho, h[0], dt)
        k2 = func(rho + 0.161 *k1, h[1], dt)
        k3 = func(rho + -0.008480655492356989*k1 + 0.335480655492357*k2, h[2], dt)
        k4 = func(rho + 2.8971530571054935*k1 -6.359448489975075*k2 + 4.3622954328695815*k3, h[3], dt)
        k5 = func(rho + 5.325864828439257*k1 -11.748883564062828*k2 + 7.4955393428898365*k3 -0.09249506636175525*k4, h[4], dt)
        k6 = func(rho + 5.86145544294642*k1 -12.92096931784711*k2 + 8.159367898576159*k3 + -0.071584973281401*k4 -0.028269050394068383*k5, h[5], dt)
        k7 = func(rho + 0.09646076681806523*k1 + 0.01*k2 + 0.4798896504144996*k3 + 1.379008574103742*k4 -3.290069515436081*k5 + 2.324710524099774*k6, h[5], dt)
    else:
        k1 = func(rho, h[0], col, dt)
        k2 = func(rho + 0.161 *k1, h[1], col, dt)
        k3 = func(rho + -0.008480655492356989*k1 + 0.335480655492357*k2, h[2], col, dt)
        k4 = func(rho + 2.8971530571054935*k1 -6.359448489975075*k2 + 4.3622954328695815*k3, h[3], col, dt)
        k5 = func(rho + 5.325864828439257*k1 -11.748883564062828*k2 + 7.4955393428898365*k3 -0.09249506636175525*k4, h[4], col, dt)
        k6 = func(rho + 5.86145544294642*k1 -12.92096931784711*k2 + 8.159367898576159*k3 + -0.071584973281401*k4 -0.028269050394068383*k5, h[5], col, dt)
        k7 = func(rho + 0.09646076681806523*k1 + 0.01*k2 + 0.4798896504144996*k3 + 1.379008574103742*k4 -3.290069515436081*k5 + 2.324710524099774*k6, h[5], col, dt)
        
    #Parameters from paper
    # rho_new = rho + 0.001780011052226*k1 + 0.000816434459657*k2 - 0.007880878010262*k3 + 0.144711007173263*k4 - 0.582357165452555*k5 + 0.458082105929187*k6 + 1./66*k7
    
    #parameters from Julia code
    rho_new = rho + 0.09468075576583945*k1 + 0.009183565540343254*k2 + 0.4877705284247616*k3 + 1.234297566930479*k4 -2.7077123499835256*k5 + 1.866628418170587*k6 + 1./66*k7
    return rho_new

def lindblad_step(rho, h, col_ops, dt):
    del_rho = -1j * commutator(h, rho)
    for col in col_ops:
        del_rho += tf.matmul(tf.matmul(col, rho), tf.transpose(col, conjugate=True))
        del_rho -= 0.5 * anticommutator(
            tf.matmul(tf.transpose(col, conjugate=True), col), rho
        )
    return del_rho * dt


def commutator(A, B):
    return tf.matmul(A, B) - tf.matmul(B, A)


def anticommutator(A, B):
    return tf.matmul(A, B) + tf.matmul(B, A)

def interpolateSignal(ts, sig, interpolate_res):
    dt = ts[1] - ts[0]
    if interpolate_res == -5:
        ts = tf.cast(ts, dtype=tf.float64)
        dt = ts[1] - ts[0]
        ts_interp = tf.concat([ts, ts+1./5*dt, ts+3./10*dt, ts+4./5*dt, ts+8./9*dt, ts+dt], axis=0)
        ts_interp = tf.sort(ts_interp)
    elif interpolate_res == -6:
        ts = tf.cast(ts, dtype=tf.float64)
        dt = ts[1] - ts[0]
        ts_interp = tf.concat([ts, ts+0.161*dt, ts+0.327*dt, ts+0.9*dt, ts+0.9800255409045097*dt, ts+dt], axis=0)
        ts_interp = tf.sort(ts_interp)
    else:
        ts_interp = tf.linspace(ts[0], ts[-1] + dt, tf.shape(ts)[0] * interpolate_res + 1)
    return tfp.math.interp_regular_1d_grid(
        ts_interp,
        ts[0],
        ts[-1],
        sig,
        fill_value="extrapolate"
    )


@state_deco
def stochastic_schrodinger_rk4(
    model: Model,
    generator: Generator, 
    instruction: Instruction,
    collapse_ops: tf.Tensor, 
    psi_init: tf.Tensor,
    L_dag_L: tf.Tensor,
    plist: tf.Tensor,
    solver="rk4",
) -> Dict:

    if solver == "rk4":
        interpolate_res = 2
    elif solver == "rk38":
        interpolate_res = 3
    elif solver == "rk5":
        interpolate_res = -5 # Fixing this a random number for now
    elif solver == "Tsit5":
        interpolate_res = -6 # Fixing this a random number for now

    hs_of_t_ts = Hs_of_t(model, generator, instruction, L_dag_L=L_dag_L, interpolate_res=interpolate_res) 
    hs = hs_of_t_ts["Hs"]
    ts = hs_of_t_ts["ts"]
    dt = hs_of_t_ts["dt"]

    psi_list = propagate_stochastic_lind(
                        model, 
                        hs, 
                        collapse_ops, 
                        psi_init, 
                        ts, 
                        dt, 
                        L_dag_L, 
                        plist, 
                        solver=solver
    )
    return {"states":psi_list, "ts": ts}


def propagate_stochastic_lind(model, hs, collapse_ops, psi_init, ts, dt, L_dag_L, plist, solver="rk4"):
    psi = psi_init
    psi_list = tf.TensorArray(
                    tf.complex128,
                    size=ts.shape[0],
                    dynamic_size=False, 
                    infer_shape=False
    )

    for index in tf.range(ts.shape[0]):
        coherent_ev_flag = 1
        counter = 0
        col_flags = []
        col_ops = []
        for key in model.subsystems:
            time1 = plist[counter][0][index]
            time2 = plist[counter][1][index]
            time_temp = plist[counter][2][index]
            col_flags.append([time1, time2, time_temp])

            relax_op = collapse_ops[counter][0]
            dec_op = collapse_ops[counter][1]
            temp_op = collapse_ops[counter][2]
            col_ops.append([relax_op, dec_op, temp_op])

            coherent_ev_flag = coherent_ev_flag * (1 - time1) * (1 - time2) * (1 - time_temp)

            counter += 1
        
        if solver == "rk38":
            h = tf.slice(hs, [3*index, 0, 0], [4, hs.shape[1], hs.shape[2]])
        elif solver == "rk5":
            h = tf.slice(hs, [6*index, 0, 0], [6, hs.shape[1], hs.shape[2]])
        elif solver == "Tsit5":
            h = tf.slice(hs, [6*index, 0, 0], [6, hs.shape[1], hs.shape[2]])
        else:
            h = tf.slice(hs, [2*index, 0, 0], [3, hs.shape[1], hs.shape[2]])
        psi = stochastic_lind_traj(h, psi, dt, col_ops, coherent_ev_flag, col_flags, solver=solver)
        psi_list = psi_list.write(index, psi)
    
    return psi_list.stack()

def schrodinger_step(psi, h, dt):
    return -1j*tf.matmul(h, psi)*dt


def stochastic_lind_traj(h, psi, dt, col_ops, coherent_ev_flag, col_flags, solver="rk4"):
    """
    Calculates the single time step lindbladian evoultion
    of a state vector.

    Parameters:
    h: Hamiltonian at given time step
    psi: state vector
    time1, time2: 1 iff the the relaxation, decoherence operators
        are to be applied
    relax_op: relaxion operator
    dec_op: decoherence operator
    """
    
    if coherent_ev_flag == 1:
        if solver == "rk38":
            psi_new = rk38_step_lind(schrodinger_step, psi, h, dt, col=None)
            #psi_new = psi_new / tf.linalg.norm(psi_new)
        elif solver == "rk5":
            psi_new = rk5_dopri_step_lind(schrodinger_step, psi, h, dt, col=None)
            #psi_new = psi_new / tf.linalg.norm(psi_new)
        elif solver == "Tsit5":
            psi_new = Tsit5_step_lind(schrodinger_step, psi, h, dt, col=None)
            #psi_new = psi_new / tf.linalg.norm(psi_new)
        else:
            psi_new = rk4_step_lind(schrodinger_step, psi, h, dt, col=None)
            #psi_new = psi_new / tf.linalg.norm(psi_new)
        
        return psi_new

    else:
        print("Collapse")
        psi_new = psi
        for i in range(len(col_flags)):
            counter = 0
            for j in range(3):
                if col_flags[i][j] == 1:
                    psi_new = tf.linalg.matmul(col_ops[i][j], psi)
                    psi_new = psi_new/tf.linalg.norm(psi_new)
                counter += 1

        return psi_new


@state_deco
def schrodinger_rk4(
    model: Model,
    gen: Generator,
    instr: Instruction,
    init_state=None,
    solver="rk4",
    renormalize_step=None
) -> Dict:

    if solver == "rk4":
        interpolate_res = 2
    elif solver == "rk38":
        interpolate_res = 3
    elif solver == "rk5":
        interpolate_res = -5 # Fixing this a random number for now
    elif solver == "Tsit5":
        interpolate_res = -6 # Fixing this a random number for now

    Hs_dict = Hs_of_t(model, gen, instr, interpolate_res=interpolate_res)
    Hs = Hs_dict["Hs"]
    ts = Hs_dict["ts"]
    dt = Hs_dict["dt"]

    psi_list = tf.TensorArray(
                    tf.complex128, 
                    size=ts.shape[0], 
                    dynamic_size=False, 
                    infer_shape=False
    )
    psi_t = init_state
    for index in tf.range(ts.shape[0]):        
        if solver =="rk38":
            h = tf.slice(Hs, [3*index, 0, 0], [4, Hs.shape[1], Hs.shape[2]])
            psi_t = rk38_step_lind(schrodinger_step, psi_t, h, dt, col=None)
        elif solver == "rk5":
            h = tf.slice(Hs, [6*index, 0, 0], [6, Hs.shape[1], Hs.shape[2]])
            psi_t = rk5_dopri_step_lind(schrodinger_step, psi_t, h, dt, col=None)
        elif solver == "Tsit5":
            h = tf.slice(Hs, [6*index, 0, 0], [6, Hs.shape[1], Hs.shape[2]])
            psi_t = Tsit5_step_lind(schrodinger_step, psi_t, h, dt, col=None)
        else:
            h = tf.slice(Hs, [2*index, 0, 0], [3, Hs.shape[1], Hs.shape[2]])
            psi_t = rk4_step_lind(schrodinger_step, psi_t, h, dt, col=None)

        if renormalize_step != None:
            if index%renormalize_step == 0:
                psi_t = psi_t/tf.linalg.norm(psi_t)
        
        psi_list = psi_list.write(index, psi_t)
    psi_list = psi_list.stack()

    return {"states": psi_list, "ts": ts}


@state_deco
def vonNeumann_rk4(
    model: Model,
    gen: Generator,
    instr: Instruction,
    init_state=None,
    solver="rk4"
) -> Dict:

    if solver == "rk4":
        interpolate_res = 2
    elif solver == "rk38":
        interpolate_res = 3
    elif solver == "rk5":
        interpolate_res = -5 # Fixing this a random number for now
    elif solver == "Tsit5":
        interpolate_res = -6 # Fixing this a random number for now

    Hs_dict = Hs_of_t(model, gen, instr, interpolate_res=interpolate_res)
    Hs = Hs_dict["Hs"]
    ts = Hs_dict["ts"]
    dt = Hs_dict["dt"]

    rho_list = tf.TensorArray(
                    tf.complex128, 
                    size=ts.shape[0], 
                    dynamic_size=False, 
                    infer_shape=False
    )
    rho_t = init_state
    for index in tf.range(ts.shape[0]):
        if solver =="rk38":
            h = tf.slice(Hs, [3*index, 0, 0], [4, Hs.shape[1], Hs.shape[2]])
            rho_t = rk38_step_lind(vonNeumann_step, rho_t, h, dt, col=None)
        elif solver == "rk5":
            h = tf.slice(Hs, [6*index, 0, 0], [6, Hs.shape[1], Hs.shape[2]])
            rho_t = rk5_dopri_step_lind(vonNeumann_step, rho_t, h, dt, col=None)
        elif solver == "Tsit5":
            h = tf.slice(Hs, [6*index, 0, 0], [6, Hs.shape[1], Hs.shape[2]])
            rho_t = Tsit5_step_lind(vonNeumann_step, rho_t, h, dt, col=None)
        else:
            h = tf.slice(Hs, [2*index, 0, 0], [3, Hs.shape[1], Hs.shape[2]])
            rho_t = rk4_step_lind(vonNeumann_step, rho_t, h, dt, col=None)

        rho_list = rho_list.write(index, rho_t)
    rho_list = rho_list.stack()

    return {"states": rho_list, "ts": ts}


def vonNeumann_step(rho, h, dt):
    return -1j * commutator(h, rho)*dt

#@state_deco
#def stochastic_schrodinger_rk4(
#    model: Model,
#    generator: Generator, 
#    instruction: Instruction,
#    collapse_ops: tf.Tensor, 
#    psi_init: tf.Tensor,
#    L_dag_L: tf.Tensor,
#    plist: tf.Tensor,
#    solver="rk4",
#) -> Dict:
#
#    if solver == "rk4":
#        interpolate_res = 2
#    elif solver == "rk38":
#        interpolate_res = 3
#
#    hs_of_t_ts = Hs_of_t(model, generator, instruction, L_dag_L=L_dag_L, interpolate_res=interpolate_res) 
#    hs = hs_of_t_ts["Hs"]
#    ts = hs_of_t_ts["ts"]
#    dt = hs_of_t_ts["dt"]
#
#    psi_list = propagate_stochastic_lind(
#                        model, 
#                        hs, 
#                        collapse_ops, 
#                        psi_init, 
#                        ts, 
#                        dt, 
#                        L_dag_L, 
#                        plist, 
#                        solver=solver
#    )
#    return {"states":psi_list, "ts": ts}
#
#
#def propagate_stochastic_lind(model, hs, collapse_ops, psi_init, ts, dt, L_dag_L, plist, solver="rk4"):
#    psi = psi_init
#    psi_list = tf.TensorArray(
#                    tf.complex128,
#                    size=ts.shape[0],
#                    dynamic_size=False, 
#                    infer_shape=False
#    )
#
#    for index in tf.range(ts.shape[0]):
#        relax_op_list = []
#        dec_op_list = []
#        temp_op_list = []
#        coherent_ev_flag = 1
#        counter = 0
#        for key in model.subsystems:
#            time1 = plist[counter][0][index]
#            time2 = plist[counter][1][index]
#            time_temp = plist[counter][2][index]
#
#            relax_op = time1 * collapse_ops[counter][0]
#            dec_op = time2 * collapse_ops[counter][1]
#            temp_op = time_temp * collapse_ops[counter][2]
#
#            relax_op_list.append(relax_op)
#            dec_op_list.append(dec_op)
#            temp_op_list.append(temp_op)
#            
#            coherent_ev_flag = coherent_ev_flag * (1 - time1) * (1 - time2) * (1 - time_temp)
#
#            counter += 1
#        
#        if solver == "rk38":
#            h = tf.slice(hs, [3*index, 0, 0], [4, hs.shape[1], hs.shape[2]])
#        else:
#            h = tf.slice(hs, [2*index, 0, 0], [3, hs.shape[1], hs.shape[2]])
#        psi = rk4_lind_traj(h, psi, dt, relax_op_list, dec_op_list, temp_op_list, coherent_ev_flag, L_dag_L, solver=solver)
#        psi_list = psi_list.write(index, psi)
#    
#    return psi_list.stack()
#
#def stochastic_step(psi, h, dt):
#    return - 1j*tf.matmul(h, psi)*dt
#
#
#def rk4_lind_traj(h, psi, dt, relax_ops, dec_ops, temp_ops, coherent_ev_flag, L_dag_L, solver="rk4"):
#    """
#    Calculates the single time step lindbladian evoultion
#    of a state vector.
#    Parameters:
#    h: Hamiltonian at given time step
#    psi: state vector
#    time1, time2: 1 iff the the relaxation, decoherence operators
#        are to be applied
#    relax_op: relaxion operator
#    dec_op: decoherence operator
#    """
#    # TODO - check for normalization of the states
#    # TODO - What happens if two of them become one at the same time
#
#    pjk = []
#    for values in L_dag_L:
#        del_pk_T1 = tf.matmul(
#                tf.transpose(psi, conjugate=True),
#                tf.matmul(
#                    values[0], psi
#                )
#        )[0][0]
#        del_pk_T2 = tf.matmul(
#                tf.transpose(psi, conjugate=True),
#                tf.matmul(
#                    values[1], psi
#                )
#        )[0][0]
#        del_pk_Temp = tf.matmul(
#                tf.transpose(psi, conjugate=True),
#                tf.matmul(
#                    values[2], psi
#                )
#        )[0][0]
#
#        pjk.append([del_pk_T1, del_pk_T2, del_pk_Temp])
#
#    pj = tf.reduce_sum(pjk) * dt
#
#    if solver == "rk38":
#        psi_new = coherent_ev_flag * rk38_step_lind(stochastic_step, psi, h, dt) * (1/tf.sqrt(1 - pj))
#    else:
#        psi_new = coherent_ev_flag * rk4_step_lind(stochastic_step, psi, h, dt) * (1/tf.sqrt(1 - pj))
#
#    # TODO - check if the condition below is correct. I have used this just for the time being.
#    if tf.reduce_prod(pjk) != 0:
#        for i in range(len(relax_ops)):
#            psi_new = (
#                        psi_new 
#                        + tf.linalg.matmul(relax_ops[i],psi) * tf.sqrt(1/pjk[i][0])
#                        + tf.linalg.matmul(dec_ops[i],psi) *   tf.sqrt(1/pjk[i][1])
#                        + tf.linalg.matmul(temp_ops[i],psi) *  tf.sqrt(1/pjk[i][2])
#                    )
#    return psi_new