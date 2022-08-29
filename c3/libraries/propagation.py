"A library for propagators and closely related functions"
import numpy as np
import tensorflow as tf
from typing import Dict
from c3.model import Model
from c3.generator.generator import Generator
from c3.signal.gates import Instruction
from c3.utils.tf_utils import (
    tf_kron,
    tf_matmul_left,
    tf_matmul_n,
    tf_spre,
    tf_spost,
    commutator,
    anticommutator,
    calculate_expectation_value,
)
from scipy import interpolate, integrate

unitary_provider = dict()
state_provider = dict()
solver_dict = dict()
step_dict = dict()

# Dictionary specifying the slice length for a dt for every solver
# the first element is the interpolation resolution
# the second element is the number of arguments per time step
# the third element is written in tf_utils.interpolate_signal according to corresponding Tableau
solver_slicing = {
    "rk4": [2, 3, 2],
    "rk38": [3, 4, 3],
    "rk5": [6, 6, -1],
    "tsit5": [6, 6, -2],
    "vern7": [9, 9, -3],
    "vern8": [12, 12, -4],
}


def step_vonNeumann_psi(psi, h, dt):
    return -1j * dt * tf.linalg.matvec(h, psi)


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


def solver_deco(func):
    """
    Decorator for making registry of solvers
    """
    solver_dict[str(func.__name__)] = func
    return func


def step_deco(func):
    """
    Decorator for making registry of solvers
    """
    step_dict[str(func.__name__)] = func
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
def rk4_unitary(
    model: Model, gen: Generator, instr: Instruction, init_state=None
) -> Dict:
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
        batch_size = tf.constant(len(ts), tf.int32)
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
def ode_solver(
    model: Model, gen: Generator, instr: Instruction, init_state, solver, step_function
) -> Dict:

    signal = gen.generate_signals(instr)

    if model.lindbladian:
        col = model.get_Lindbladians()
        step_function = "lindblad"
    else:
        col = None

    interpolate_res = solver_slicing[solver][2]

    Hs_dict = model.Hs_of_t(signal, interpolate_res=interpolate_res)
    Hs = Hs_dict["Hs"]
    ts = Hs_dict["ts"]
    dt = Hs_dict["dt"]

    state_list = tf.TensorArray(
        tf.complex128, size=ts.shape[0], dynamic_size=False, infer_shape=False
    )
    state_t = init_state
    start = solver_slicing[solver][0]
    stop = solver_slicing[solver][1]
    ode_step = step_dict[step_function]
    solver_function = solver_dict[solver]
    for index in tf.range(ts.shape[0]):
        h = tf.slice(Hs, [start * index, 0, 0], [stop, Hs.shape[1], Hs.shape[2]])
        state_t = solver_function(ode_step, state_t, h, dt, col=col)
        state_list = state_list.write(index, state_t)

    states = state_list.stack()

    return {"states": states, "ts": ts}


@state_deco
def ode_solver_final_state(
    model: Model, gen: Generator, instr: Instruction, init_state, solver, step_function
) -> Dict:

    signal = gen.generate_signals(instr)

    if model.lindbladian:
        col = model.get_Lindbladians()
        step_function = "lindblad"
    else:
        col = None

    interpolate_res = solver_slicing[solver][2]

    Hs_dict = model.Hs_of_t(signal, interpolate_res=interpolate_res)
    Hs = Hs_dict["Hs"]
    ts = Hs_dict["ts"]
    dt = Hs_dict["dt"]

    state_t = init_state
    start = solver_slicing[solver][0]
    stop = solver_slicing[solver][1]
    ode_step = step_dict[step_function]
    solver_function = solver_dict[solver]
    for index in tf.range(ts.shape[0]):
        h = tf.slice(Hs, [start * index, 0, 0], [stop, Hs.shape[1], Hs.shape[2]])
        state_t = solver_function(ode_step, state_t, h, dt, col=col)

    return {"states": state_t, "ts": ts}


@state_deco
def scipy_integrate(
    model: Model, gen: Generator, instr: Instruction, init_state, solver, step_function
) -> Dict:

    solvers = ["vode", "zvode", "lsoda", "dopri5", "dop853"]
    if solver not in solvers:
        raise Exception(f"solver not found. Available solvers are {solvers}")

    signal = gen.generate_signals(instr)
    h0, hctrls = model.get_Hamiltonians()

    ts_list = []
    signals = []
    hks = []
    for key in signal:
        ts_list.append(signal[key]["ts"])
        signals.append(signal[key]["values"])
        hks.append(hctrls[key])

    ts = tf.reduce_mean(ts_list, axis=0)
    dt = ts[1] - ts[0]
    nsteps = ts.shape[0]

    signal_functions = []
    for sig in signals:
        signal_fun = interpolate.interp1d(ts, sig, fill_value="extrapolate")
        signal_functions.append(signal_fun)

    if model.lindbladian:
        collapse = model.get_Lindbladians()
    else:
        collapse = None

    states_list = []

    def ham_func(t):
        ham = h0
        i = 0
        for sig_fun in signal_functions:
            ham += sig_fun(t) * hks[i]
            i += 1
        return ham

    if model.lindbladian:

        def ode_func(t, psi):
            del_rho = -1j * commutator(ham_func(t), psi)
            for col in collapse:
                del_rho += tf.matmul(
                    tf.matmul(col, psi), tf.transpose(col, conjugate=True)
                )
                del_rho -= 0.5 * anticommutator(
                    tf.matmul(tf.transpose(col, conjugate=True), col), psi
                )
            return del_rho * dt

    else:

        def ode_func(t, psi):
            return -1j * tf.linalg.matvec(ham_func(t), psi.T)

    r = integrate.ode(ode_func)
    r.set_integrator(solver, method="bdf", nsteps=nsteps)
    r.set_initial_value(init_state, ts[0])

    for i in range(nsteps):
        states_list.append(r.integrate(r.t + dt))

    states = tf.convert_to_tensor(states_list, dtype=tf.complex128)
    ts = tf.cast(ts, dtype=tf.complex128)

    return {"states": states, "ts": ts}


def rk_using_butcher_tableau(func, rho, h, dt, tableau, col=None):
    a, b, c = tableau
    ks = []
    for i in range(len(b)):
        y_new = rho + tf.reduce_sum(tf.multiply(a[i], ks))
        k_new = func(y_new, h[i], dt, col)
        ks.append(k_new)
    rho_new = rho + tf.reduce_sum(tf.multiply(b, ks), axis=0)
    return rho_new


@solver_deco
def rk4(func, rho, h, dt, col=None):
    k1 = func(rho, h[0], dt, col)
    k2 = func(rho + k1 / 2.0, h[1], dt, col)
    k3 = func(rho + k2 / 2.0, h[1], dt, col)
    k4 = func(rho + k3, h[2], dt, col)
    rho_new = rho + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return rho_new


@solver_deco
def rk38(func, rho, h, dt, col=None):
    k1 = func(rho, h[0], dt, col)
    k2 = func(rho + k1 / 3.0, h[1], dt, col)
    k3 = func(rho + (-k1 / 3.0) + k2, h[2], dt, col)
    k4 = func(rho + k1 - k2 + k3, h[3], dt, col)
    rho_new = rho + (k1 + 3 * k2 + 3 * k3 + k4) / 8.0
    return rho_new


@solver_deco
def rk5(func, rho, h, dt, col=None):
    k1 = func(rho, h[0], dt, col)
    k2 = func(rho + 1.0 / 5 * k1, h[1], dt, col)
    k3 = func(rho + 3.0 / 40 * k1 + 9.0 / 40 * k2, h[2], dt, col)
    k4 = func(rho + 44.0 / 45 * k1 - 56.0 / 15 * k2 + 32.0 / 9 * k3, h[3], dt, col)
    k5 = func(
        rho
        + 19372.0 / 6561 * k1
        - 25360.0 / 2187 * k2
        + 64448.0 / 6561 * k3
        - 212.0 / 729 * k4,
        h[4],
        dt,
        col,
    )
    k6 = func(
        rho
        + 9017.0 / 3168 * k1
        - 355.0 / 33 * k2
        + 46732.0 / 5247 * k3
        + 49.0 / 176 * k4
        - 5103.0 / 18656 * k5,
        h[5],
        dt,
        col,
    )
    k7 = func(
        rho
        + 35.0 / 384 * k1
        + 500.0 / 1113 * k3
        + 125.0 / 192 * k4
        - 2187.0 / 6784 * k5
        + 11.0 / 84 * k6,
        h[5],
        dt,
        col,
    )

    rho_new = (
        rho
        + 5179.0 / 57600 * k1
        + 7571.0 / 16695 * k3
        + 393.0 / 640 * k4
        - 92097.0 / 339200 * k5
        + 187.0 / 2100 * k6
        + 1.0 / 40 * k7
    )
    return rho_new


@solver_deco
def tsit5(func, rho, h, dt, col=None):
    k1 = func(rho, h[0], dt, col)
    k2 = func(rho + 0.161 * k1, h[1], dt, col)
    k3 = func(rho + -0.008480655492356989 * k1 + 0.335480655492357 * k2, h[2], dt, col)
    k4 = func(
        rho
        + 2.8971530571054935 * k1
        - 6.359448489975075 * k2
        + 4.3622954328695815 * k3,
        h[3],
        dt,
        col,
    )
    k5 = func(
        rho
        + 5.325864828439257 * k1
        - 11.748883564062828 * k2
        + 7.4955393428898365 * k3
        - 0.09249506636175525 * k4,
        h[4],
        dt,
        col,
    )
    k6 = func(
        rho
        + 5.86145544294642 * k1
        - 12.92096931784711 * k2
        + 8.159367898576159 * k3
        + -0.071584973281401 * k4
        - 0.028269050394068383 * k5,
        h[5],
        dt,
        col,
    )
    k7 = func(
        rho
        + 0.09646076681806523 * k1
        + 0.01 * k2
        + 0.4798896504144996 * k3
        + 1.379008574103742 * k4
        - 3.290069515436081 * k5
        + 2.324710524099774 * k6,
        h[5],
        dt,
        col,
    )
    rho_new = (
        rho
        + 0.09468075576583945 * k1
        + 0.009183565540343254 * k2
        + 0.4877705284247616 * k3
        + 1.234297566930479 * k4
        - 2.7077123499835256 * k5
        + 1.866628418170587 * k6
        + 1.0 / 66 * k7
    )
    return rho_new


@solver_deco
def vern7(func, rho, h, dt, col=None):
    k1 = func(rho, h[0], dt, col)
    k2 = func(rho + (1 / 200) * k1, h[1], dt, col)
    k3 = func(rho + (-4361 / 4050) * k1 + (2401 / 2025) * k2, h[2], dt, col)
    k4 = func(
        rho + (49 / 1200) * k1 + (49 / 400) * k3,
        h[3],
        dt,
        col,
    )
    k5 = func(
        rho
        + (2454451729 / 3841600000) * k1
        + (-9433712007 / 3841600000) * k3
        + (4364554539 / 1920800000) * k4,
        h[4],
        dt,
        col,
    )
    k6 = func(
        rho
        + (
            -6187101755456742839167388910402379177523537620
            / 2324599620333464857202963610201679332423082271
        )
        * k1
        + (
            27569888999279458303270493567994248533230000
            / 2551701010245296220859455115479340650299761
        )
        * k3
        + (
            -37368161901278864592027018689858091583238040000
            / 4473131870960004275166624817435284159975481033
        )
        * k4
        + (
            1392547243220807196190880383038194667840000000
            / 1697219131380493083996999253929006193143549863
        )
        * k5,
        h[5],
        dt,
        col,
    )
    k7 = func(
        rho
        + (11272026205260557297236918526339 / 1857697188743815510261537500000) * k1
        + (-48265918242888069 / 1953194276993750) * k3
        + (26726983360888651136155661781228 / 1308381343805114800955157615625) * k4
        + (-2090453318815827627666994432 / 1096684189897834170412307919) * k5
        + (
            1148577938985388929671582486744843844943428041509
            / 1141532118233823914568777901158338927629837500000
        )
        * k6,
        h[6],
        dt,
        col,
    )
    # k8 = func(
    #     rho
    #     + (1304457204588839386329181466225966641
    #         /108211771565488329642169667802016000) * k1
    #     + (-1990261989751005/40001418792832) * k3
    #     + (2392691599894847687194643439066780106875
    #         /58155654089143548047476915856270826016) * k4
    #     + (-1870932273351008733802814881998561250
    #         /419326053051486744762255151208232123) * k5
    #     + (1043329047173803328972823866240311074041739158858792987034783181
    #         /510851127745017966999893975119259285040213723744255237522144000) * k6
    #     + (-311918858557595100410788125/3171569057622789618800376448) * k7,
    #     h[7],
    #     dt,
    #     col,
    # )
    # k9 = func(
    #     rho
    #     + (17579784273699839132265404100877911157
    #         /1734023495717116205617154737841023480) * k1
    #     + (-18539365951217471064750/434776548575709731377) * k3
    #     + (447448655912568142291911830292656995992000
    #         /12511202807447096607487664209063950964109) * k4
    #     + (-65907597316483030274308429593905808000000
    #         /15158061430635748897861852383197382130691) * k5
    #     + (273847823027445129865693702689010278588244606493753883568739168819449761
    #         /136252034448398939768371761610231099586032870552034688235302796640584360) * k6
    #     + (694664732797172504668206847646718750
    #         /1991875650119463976442052358853258111) * k7
    #     + (-19705319055289176355560129234220800/72595753317320295604316217197876507) * k8,
    #     h[8],
    #     dt,
    #     col,
    # )
    k10 = func(
        rho
        + (-511858190895337044664743508805671 / 11367030248263048398341724647960) * k1
        + (2822037469238841750 / 15064746656776439) * k3
        + (
            -23523744880286194122061074624512868000
            / 152723005449262599342117017051789699
        )
        * k4
        + (
            10685036369693854448650967542704000000
            / 575558095977344459903303055137999707
        )
        * k5
        + (
            -6259648732772142303029374363607629515525848829303541906422993
            / 876479353814142962817551241844706205620792843316435566420120
        )
        * k6
        + (17380896627486168667542032602031250 / 13279937889697320236613879977356033)
        * k7,
        h[8],
        dt,
        col,
    )
    rho_new = (
        rho
        + (117807213929927 / 2640907728177740) * k1
        + (4758744518816629500000 / 17812069906509312711137) * k4
        + (1730775233574080000000000 / 7863520414322158392809673) * k5
        + (
            2682653613028767167314032381891560552585218935572349997
            / 12258338284789875762081637252125169126464880985167722660
        )
        * k6
        + (40977117022675781250 / 178949401077111131341) * k7
        + (2152106665253777 / 106040260335225546) * k10
    )
    return rho_new


@solver_deco
def vern8(func, rho, h, dt, col=None):
    k1 = func(rho, h[0], dt, col)
    k2 = func(rho + (1 / 20) * k1, h[1], dt, col)
    k3 = func(rho + (-7161 / 1024000) * k1 + (116281 / 1024000) * k2, h[2], dt, col)
    k4 = func(
        rho + (1023 / 25600) * k1 + (3069 / 25600) * k3,
        h[3],
        dt,
        col,
    )
    k5 = func(
        rho
        + (4202367 / 11628100) * k1
        + (-3899844 / 2907025) * k3
        + (3982992 / 2907025) * k4,
        h[4],
        dt,
        col,
    )
    k6 = func(
        rho + (5611 / 114400) * k1 + (31744 / 135025) * k4 + (923521 / 5106400) * k5,
        h[5],
        dt,
        col,
    )
    k7 = func(
        rho
        + (21173 / 343200) * k1
        + (8602624 / 76559175) * k4
        + (-26782109 / 689364000) * k5
        + (5611 / 283500) * k6,
        h[6],
        dt,
        col,
    )
    k8 = func(
        rho
        + (-1221101821869329 / 690812928000000) * k1
        + (-125 / 2) * k4
        + (-1024030607959889 / 168929280000000) * k5
        + (1501408353528689 / 265697280000000) * k6
        + (6070139212132283 / 92502016000000) * k7,
        h[7],
        dt,
        col,
    )
    k9 = func(
        rho
        + (
            -1472514264486215803881384708877264246346044433307094207829051978044531801133057155
            / 1246894801620032001157059621643986024803301558393487900440453636168046069686436608
        )
        * k1
        + (
            -5172294311085668458375175655246981230039025336933699114138315270772319372469280000
            / 124619381004809145897278630571215298365257079410236252921850936749076487132995191
        )
        * k4
        + (
            -12070679258469254807978936441733187949484571516120469966534514296406891652614970375
            / 2722031154761657221710478184531100699497284085048389015085076961673446140398628096
        )
        * k5
        + (
            780125155843893641323090552530431036567795592568497182701460674803126770111481625
            / 183110425412731972197889874507158786859226102980861859505241443073629143100805376
        )
        * k6
        + (
            664113122959911642134782135839106469928140328160577035357155340392950009492511875
            / 15178465598586248136333023107295349175279765150089078301139943253016877823170816
        )
        * k7
        + (
            10332848184452015604056836767286656859124007796970668046446015775000000
            / 1312703550036033648073834248740727914537972028638950165249582733679393783
        )
        * k8,
        h[8],
        dt,
        col,
    )
    k10 = func(
        rho
        + (
            -29055573360337415088538618442231036441314060511
            / 22674759891089577691327962602370597632000000000
        )
        * k1
        + (-20462749524591049105403365239069 / 454251913499893469596231268750) * k4
        + (-180269259803172281163724663224981097 / 38100922558256871086579832832000000)
        * k5
        + (
            21127670214172802870128286992003940810655221489
            / 4679473877997892906145822697976708633673728000
        )
        * k6
        + (
            318607235173649312405151265849660869927653414425413
            / 6714716715558965303132938072935465423910912000000
        )
        * k7
        + (212083202434519082281842245535894 / 20022426044775672563822865371173879) * k8
        + (
            -2698404929400842518721166485087129798562269848229517793703413951226714583
            / 469545674913934315077000442080871141884676035902717550325616728175875000000
        )
        * k9,
        h[9],
        dt,
        col,
    )
    k11 = func(
        rho
        + (
            -2342659845814086836951207140065609179073838476242943917
            / 1358480961351056777022231400139158760857532162795520000
        )
        * k1
        + (-996286030132538159613930889652 / 16353068885996164905464325675) * k4
        + (-26053085959256534152588089363841 / 4377552804565683061011299942400) * k5
        + (
            20980822345096760292224086794978105312644533925634933539
            / 3775889992007550803878727839115494641972212962174156800
        )
        * k6
        + (
            890722993756379186418929622095833835264322635782294899
            / 13921242001395112657501941955594013822830119803764736
        )
        * k7
        + (
            161021426143124178389075121929246710833125
            / 10997207722131034650667041364346422894371443
        )
        * k8
        + (
            300760669768102517834232497565452434946672266195876496371874262392684852243925359864884962513
            / 4655443337501346455585065336604505603760824779615521285751892810315680492364106674524398280000
        )
        * k9
        + (-31155237437111730665923206875 / 392862141594230515010338956291) * k10,
        h[10],
        dt,
        col,
    )
    k12 = func(
        rho
        + (
            -2866556991825663971778295329101033887534912787724034363
            / 868226711619262703011213925016143612030669233795338240
        )
        * k1
        + (
            -16957088714171468676387054358954754000
            / 143690415119654683326368228101570221
        )
        * k4
        + (
            -4583493974484572912949314673356033540575
            / 451957703655250747157313034270335135744
        )
        * k5
        + (
            2346305388553404258656258473446184419154740172519949575
            / 256726716407895402892744978301151486254183185289662464
        )
        * k6
        + (
            1657121559319846802171283690913610698586256573484808662625
            / 13431480411255146477259155104956093505361644432088109056
        )
        * k7
        + (
            345685379554677052215495825476969226377187500
            / 74771167436930077221667203179551347546362089
        )
        * k8
        + (
            -3205890962717072542791434312152727534008102774023210240571361570757249056167015230160352087048674542196011
            / 947569549683965814783015124451273604984657747127257615372449205973192657306017239103491074738324033259120
        )
        * k9
        + (
            40279545832706233433100438588458933210937500
            / 8896460842799482846916972126377338947215101
        )
        * k10
        + (
            -6122933601070769591613093993993358877250
            / 1050517001510235513198246721302027675953
        )
        * k11,
        h[11],
        dt,
        col,
    )
    # k13 = func(
    #     rho
    #     + (-618675905535482500672800859344538410358660153899637
    #         /203544282118214047100119475340667684874292102389760) * k1
    #     + (-4411194916804718600478400319122931000
    #         /40373053902469967450761491269633019) * k4
    #     + (-16734711409449292534539422531728520225
    #         /1801243715290088669307203927210237952) * k5
    #     + (135137519757054679098042184152749677761254751865630525
    #         /16029587794486289597771326361911895112703716593983488) * k6
    #     + (38937568367409876012548551903492196137929710431584875
    #         /340956454090191606099548798001469306974758443147264) * k7
    #     + (-6748865855011993037732355335815350667265625
    #         /7002880395717424621213565406715087764770357) * k8
    #     + (-1756005520307450928195422767042525091954178296002788308926563193523662404739779789732685671
    #         /348767814578469983605688098046186480904607278021030540735333862087061574934154942830062320) * k9
    #     + (53381024589235611084013897674181629296875
    #         /8959357584795694524874969598508592944141) * k10,
    #     h[11],
    #     dt,
    #     col,
    # )
    # For 13
    # rho_new = (
    #     rho
    #     + (10835401739407019406577/244521829356935137978320) * k1
    #     + (13908189778321895491375000/39221135527894265375640567) * k6
    #     + (73487947527027243487625000/296504045773342769773399443) * k7
    #     + (68293140641257649609375000/15353208647806945749946119) * k8
    #     + (22060647948996678611017711379974578860522018208949721559448560203338437626022142776381
    #         /1111542009262325874512959185795727215759010577565736079641376621381577236680929558640) * k9
    #     + (-547971229495642458203125000/23237214025700991642563601) * k10
    #     + (-28735456870978964189/79783493704265043693) * k13

    # )

    # For 12
    rho_new = (
        rho
        + (44901867737754616851973 / 1014046409980231013380680) * k1
        + (791638675191615279648100000 / 2235604725089973126411512319) * k6
        + (3847749490868980348119500000 / 15517045062138271618141237517) * k7
        + (-13734512432397741476562500000 / 875132892924995907746928783) * k8
        + (
            12274765470313196878428812037740635050319234276006986398294443554969616342274215316330684448207141
            / 489345147493715517650385834143510934888829280686609654482896526796523353052166757299452852166040
        )
        * k9
        + (-9798363684577739445312500000 / 308722986341456031822630699) * k10
        + (282035543183190840068750 / 12295407629873040425991) * k11
        + (-306814272936976936753 / 1299331183183744997286) * k12
    )
    return rho_new


@step_deco
def lindblad(rho, h, dt, col):
    del_rho = -1j * commutator(h, rho)
    for col in col:
        del_rho += tf.matmul(tf.matmul(col, rho), tf.transpose(col, conjugate=True))
        del_rho -= 0.5 * anticommutator(
            tf.matmul(tf.transpose(col, conjugate=True), col), rho
        )
    return del_rho * dt


@step_deco
def schrodinger(psi, h, dt, col=None):
    return -1j * tf.matmul(h, psi) * dt


@step_deco
def von_neumann(rho, h, dt, col=None):
    return -1j * commutator(h, rho) * dt


@state_deco
def stochastic_schrodinger_rk4(
    model: Model,
    gen: Generator,
    instr: Instruction,
    collapse_ops: tf.Tensor,
    init_state: tf.Tensor,
    L_dag_L: tf.Tensor,
    plist: tf.Tensor,
    solver="rk4",
) -> Dict:

    signal = gen.generate_signals(instr)
    interpolate_res = solver_slicing[solver][2]

    hs_of_t_ts = model.Hs_of_t(signal, interpolate_res=interpolate_res, L_dag_L=L_dag_L)
    Hs = hs_of_t_ts["Hs"]
    ts = hs_of_t_ts["ts"]
    dt = hs_of_t_ts["dt"]

    psi_list = propagate_stochastic_lind(
        model, Hs, collapse_ops, init_state, ts, dt, plist, L_dag_L, solver=solver
    )
    return {"states": psi_list, "ts": ts}


def propagate_stochastic_lind(
    model, hs, collapse_ops, psi_init, ts, dt, plist, L_dag_L, solver="rk4"
):

    start = solver_slicing[solver][0]
    stop = solver_slicing[solver][1]
    solver_function = solver_dict[solver]

    psi = psi_init
    psi_list = tf.TensorArray(
        tf.complex128, size=ts.shape[0], dynamic_size=False, infer_shape=False
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

            coherent_ev_flag = (
                coherent_ev_flag * (1 - time1) * (1 - time2) * (1 - time_temp)
            )

            counter += 1

        h = tf.slice(hs, [start * index, 0, 0], [stop, hs.shape[1], hs.shape[2]])

        psi = stochastic_lind_traj(
            h, psi, dt, col_ops, coherent_ev_flag, col_flags, solver_function
        )
        if tf.abs(coherent_ev_flag) == 0:
            plist = recompute_plist(model, ts.shape[0], dt, psi, L_dag_L)
        psi_list = psi_list.write(index, psi)

    return psi_list.stack()


def schrodinger_step(psi, h, dt):
    return -1j * tf.matmul(h, psi) * dt


def stochastic_lind_traj(
    h, psi, dt, col_ops, coherent_ev_flag, col_flags, solver_function
):
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
        psi_new = solver_function(schrodinger, psi, h, dt, col=None)
        psi_new = tf.math.l2_normalize(psi_new)
        return psi_new

    else:
        print("Collapse")
        psi_new = psi
        for i in range(len(col_flags)):
            counter = 0
            for j in range(3):
                if col_flags[i][j] == 1:
                    psi_new = tf.linalg.matmul(col_ops[i][j], psi)
                counter += 1

        psi_new = tf.math.l2_normalize(psi_new)
        return psi_new


def recompute_plist(model, ts_len, dt, psi, L_dag_L):
    dt = tf.cast(dt, dtype=tf.float64)
    plists = []
    counter = 0
    g = tf.random.get_global_generator()

    for key in model.subsystems:
        p_vals = []

        temp1 = g.uniform(shape=[ts_len], dtype=tf.float64)
        temp2 = g.uniform(shape=[ts_len], dtype=tf.float64)
        tempt = g.uniform(shape=[ts_len], dtype=tf.float64)

        pT1 = tf.abs(calculate_expectation_value(psi, L_dag_L[counter][0])) * dt
        pT2 = tf.abs(calculate_expectation_value(psi, L_dag_L[counter][1])) * dt
        pTemp = tf.abs(calculate_expectation_value(psi, L_dag_L[counter][2])) * dt

        p_vals.append(tf.math.floor((tf.math.sign(-temp1 + pT1) + 1) / 2))
        p_vals.append(tf.math.floor((tf.math.sign(-temp2 + pT2) + 1) / 2))
        p_vals.append(tf.math.floor((tf.math.sign(-tempt + pTemp) + 1) / 2))

        plists.append(p_vals)
        counter += 1
    return tf.cast(plists, dtype=tf.complex128)
