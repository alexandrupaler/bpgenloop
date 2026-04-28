# from quimb_message_passing import neighbors
import math
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from tqdm import tqdm

from message_passing_variables import get_general_neighborhood_variables



def expectation_log_Ti(S_i_tn, T):
    vars_set = set().union(*[set(T.inds) for T in S_i_tn.tensors])
    vars_order = sorted(vars_set)
    Z = 0.0
    num = 0.0
    for bits in np.ndindex(*(2,) * len(vars_order)):
        a = {v: b for v, b in zip(vars_order, bits)}
        w = 1.0
        for U in S_i_tn.tensors:
            w *= eval_tensor_at_assignment(U, a)
        Ti_val = eval_tensor_at_assignment(T, a)
        if Ti_val > 0:
            num += w * np.log(Ti_val)
        Z += w
    return num / Z


def calculate_l0_neighbourhoods_variables(adj, l0):
    neighbourhoods = {}
    for i in range(adj.shape[0]):
        neighbourhoods[i] = get_general_neighborhood_variables(i, l0, adj)
    return neighbourhoods


# equation B5 derivation
def compute_energy_exact_general(tn, Z):
    energy = 0.0

    for i in tqdm(range(len(tn.tensors))):
        T_i__tn = tn.select([f"N{i}"])
        T = T_i__tn.tensors[0]

        probability = tn.contract(output_inds=(T.inds))
        probability = qtn.Tensor(data=probability.data, inds=probability.inds, tags={})

        T_log = qtn.Tensor(data=np.log(T.data), inds=T.inds, tags={})

        vars_set = set().union(*[set(T.inds) for T in [probability]])
        vars_order = sorted(vars_set)
        sum_trace = 0
        # Z_cal = 0
        for bits in np.ndindex(*(2,) * len(vars_order)):
            a = {v: b for v, b in zip(vars_order, bits)}
            prob = eval_tensor_at_assignment(probability, a)
            t_val = eval_tensor_at_assignment(T_log, a)
            sum_trace += prob * t_val
            # Z_cal += prob
        energy -= sum_trace

    return energy / Z


# equation B5 derivation
def compute_energy_exact(size, tn):
    energy = 0.0

    for x in range(size):
        for y in range(size):
            T_i__tn = tn.select([f"T_{x}_{y}"])
            T = T_i__tn.tensors[0]

            probability = tn.contract(output_inds=(T.inds))
            probability = qtn.Tensor(
                data=probability.data, inds=probability.inds, tags={}
            )

            T_log = qtn.Tensor(data=np.log(T.data), inds=T.inds, tags={})

            vars_set = set().union(*[set(T.inds) for T in [probability]])
            vars_order = sorted(vars_set)
            sum_trace = 0
            Z_cal = 0
            for bits in np.ndindex(*(2,) * len(vars_order)):
                a = {v: b for v, b in zip(vars_order, bits)}
                prob = eval_tensor_at_assignment(probability, a)
                t_val = eval_tensor_at_assignment(T_log, a)
                sum_trace += prob * t_val
                Z_cal += prob
            energy -= sum_trace / Z_cal
    return energy


# equation B5 derivation
def compute_energy_exact_new(tn, Z):
    energy = 0.0
    for i in tqdm(range(len(tn.tensors))):
        T_i__tn = tn.select([f"N{i}"])
        T = T_i__tn.tensors[0]

        assert len(T_i__tn.tensors) == 1
        # S_i without T_i, it is added later
        S_i__tn = tn
        # S_i/j

        trace_term_tn = S_i__tn.copy()

        T_log = qtn.Tensor(data=np.log(T.data), inds=T.inds, tags={})
        trace_term_tn.add(T_log)

        trace_term = trace_term_tn.contract(output_inds=())

        energy -= trace_term / Z
    #   energy -= trace_term_explicit
    return energy


def compute_energy_general(tn, messages, neighbourhoods_l0):
    energy = 0.0
    for i in range(len(tn.tensors)):
        T_i__tn = tn.select([f"N{i}"])
        T = T_i__tn.tensors[0]
        # S_i without T_i, it is added later
        S_i = neighbourhoods_l0[i]
        S_i_tags = [f"N{j}" for j in S_i]
        S_i__tn = tn.select(S_i_tags, which="any")
        # S_i/j

        for k in set(S_i) - set([i]):
            S_i__tn |= messages[(k, i)]

        trace_term_tn = S_i__tn.copy()

        #    trace_term_explicit = expectation_log_Ti(trace_term_tn, T_i__tn.tensors[0])

        # normalization constant
        Z_Ti = S_i__tn.contract()

        T_log = qtn.Tensor(data=np.log(T.data), inds=T.inds, tags={})
        trace_term_tn.add(T_log)

        trace_term = trace_term_tn.contract(output_inds=())

        if Z_Ti != 0:
            energy -= trace_term / Z_Ti
    #   energy -= trace_term_explicit
    return energy


def compute_energy_general_variables(tn, messages, neighbourhoods_l0):
    energy = 0.0
    for i in range(len(tn.tensors)):
        T_i__tn = tn.select([f"N{i}"])
        T = T_i__tn.tensors[0]
        # S_i without T_i, it is added later
        S_i_tensors, _ = neighbourhoods_l0[i]

        S_i__tn = qtn.TensorNetwork()

        for tensor in S_i_tensors - {i}:
            T_k = tn.select(f"N{tensor}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(tensor, i)], T_k])
            S_i__tn.add(msg_and_tensor.contract())

        S_i__tn.add(T_i__tn)

        trace_term_tn = S_i__tn.copy()

        #    trace_term_explicit = expectation_log_Ti(trace_term_tn, T_i__tn.tensors[0])

        # normalization constant
        Z_Ti = S_i__tn.contract()

        T_log = qtn.Tensor(data=np.log(T.data), inds=T.inds, tags={})
        trace_term_tn.add(T_log)

        trace_term = trace_term_tn.contract(output_inds=())

        if Z_Ti != 0:
            energy -= trace_term / Z_Ti
    #   energy -= trace_term_explicit
    return energy



def compute_marginals_exact(size, Z, tn):
    marginals = {}
    for i in range(size):
        for j in range(size):
            T_i__tn = tn.select([f"T_{i}_{j}"])
            T = T_i__tn.tensors[0]

            output = tn.contract(output_inds=(T.inds))
            output.modify(tags={f"Marginal_{i}_{j}"}, data=output.data / Z)
            marginals[(i, j)] = output

    return marginals


def compute_marginals_exact_general(tn, Z):
    marginals = {}
    for i in tqdm(range(len(tn.tensors))):
        T_i__tn = tn.select([f"N{i}"])
        T = T_i__tn.tensors[0]
        output = tn.contract(output_inds=(T.inds))
        output.modify(tags={f"Marginal_{i}"}, data=output.data / Z)
        marginals[i] = output

    return marginals


#
def compute_marginals_general(tn, messages, N_l0):
    marginals = {}
    for i in range(len(tn.tensors)):
        N_i = N_l0[i]

        local_tn = qtn.TensorNetwork([])

        for k in set(N_i) - set([i]):
            local_tn |= messages[(k, i)]

        S_i_tags = [f"N{i}" for i in N_i]

        S_i_tn = tn.select(S_i_tags, which="any")

        local_tn.add(S_i_tn)

        T = S_i_tn.select(f"N{i}").tensors[0]

        output = S_i_tn.contract(output_inds=(T.inds))
        output.modify(tags={f"Marginal_{i}"}, data=output.data / np.sum(output.data))

        marginals[i] = output

    return marginals


def compute_marginals_general_variables(tn, messages, N_l0_tensors):
    marginals = {}
    for i in tqdm(range(len(tn.tensors))):
        S_i = N_l0_tensors[i]

        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        T = T_i__tn.tensors[0]
        local_tn.add(T_i__tn)
        output = local_tn.contract(output_inds=(T.inds))
        output.modify(tags={f"Marginal_{i}"}, data=output.data / np.sum(output.data))

        marginals[i] = output

    return marginals


def tensors_that_share_neighbourhood(size, blockSize):
    halfBlockSize = math.floor(blockSize / 2)
    neighbours = []
    for xi in range(size):
        for yi in range(size):
            for xj in range(xi - halfBlockSize, xi + halfBlockSize + 1, 1):
                for yj in range(yi - halfBlockSize, yi + halfBlockSize + 1, 1):
                    if (
                        xj >= 0
                        and xj < size
                        and yj >= 0
                        and yj < size
                        and (xi, yi) != (xj, yj)
                        # other way round
                        and ((xj, yj), (xi, yi)) not in neighbours
                    ):
                        neighbours.append(((xi, yi), (xj, yj)))
    return neighbours


def tensors_that_share_edge(size):
    neighbours = []
    for xi in range(size):
        for yi in range(size):
            for x_d, y_d in [(1, 0), (0, 1)]:
                if xi + x_d < size and yi + y_d < size:
                    neighbours.append(((xi, yi), (xi + x_d, yi + y_d)))

    return neighbours


def connected(i, j):
    (xi, yi) = i
    (xj, yj) = j
    return abs(xi - xj) + abs(yi - yj) == 1



def allNeighbours(neighbourhoods_l0):
    all_neighbours = []

    for i, N_i in neighbourhoods_l0.items():
        for j in N_i - {i}:
            if i < j:
                all_neighbours.append((i, j))
    return all_neighbours


def factors_general(neighbourhoods_l0, nearest_neighbours):
    w = {}
    all_neighbours = allNeighbours(neighbourhoods_l0=neighbourhoods_l0)
    for i in range(len(neighbourhoods_l0.items())):
        sum = 0
        for j, k in all_neighbours:
            overlap_jk = neighbourhoods_l0[j] & neighbourhoods_l0[k]
            if i in overlap_jk:
                sum += 1 / math.comb(len(overlap_jk), 2)
        w[i] = 1 - sum

    c = {}
    for i in range(len(neighbourhoods_l0.items())):
        for j in range(len(neighbourhoods_l0.items())):
            if i >= j:
                continue
            sum = 0
            for k, m in all_neighbours:
                overlap_km = neighbourhoods_l0[k] & neighbourhoods_l0[m]
                if i in overlap_km and j in overlap_km and i in nearest_neighbours[j]:
                    sum += 1 / math.comb(len(overlap_km), 2)

            c[(i, j)] = 1 - (w[i] + w[j]) - sum
    print("factors calculated")
    return w, c


def factors_general_variables(N_l0_tensors, N_l0_variables, NN_tensors, NN_variables):
    w = {}
    all_neighbours = allNeighbours(neighbourhoods_l0=N_l0_tensors)

    for i in range(len(N_l0_tensors.items())):
        sum = 0
        for j, k in all_neighbours:
            overlap_jk = N_l0_tensors[j] & N_l0_tensors[k]
            if i in overlap_jk:
                sum += 1 / math.comb(len(overlap_jk), 2)
        w[i] = 1 - sum

    c = {}
    for i in range(len(N_l0_tensors.items())):
        for j in range(len(N_l0_tensors.items())):
            if i >= j:
                continue
            sum = 0
            for k, m in all_neighbours:
                overlap_km = N_l0_variables[k] & N_l0_variables[m]
                overlap_km_len_tensors = len(N_l0_tensors[k] & N_l0_tensors[m])
                if (i, j) in overlap_km:
                    sum += 1 / math.comb(overlap_km_len_tensors, 2)

            c[(i, j)] = 1 - (w[i] + w[j]) - sum
    print("factors calculated")
    return w, c


def normalize(arr):
    """Normalize a nonnegative array to sum to 1 (in-place safe: returns new array)."""
    s = arr.sum()
    if s <= 0:
        raise ValueError("Array sum is nonpositive; check your contraction.")
    return arr / s


def eval_tensor_at_assignment(
    t: qu.tensor.tensor_core.Tensor, assignment: dict
) -> float:
    """
    Evaluate a quimb Tensor at a fixed assignment of (some or all) of its inds.
    Returns the resulting scalar (float).
    - assignment: dict like {'i': 0, 'j': 1, ...}
    """
    # Fix the relevant indices using integer selection (no contraction!)
    # t.isel returns a reduced Tensor; if all inds are fixed, it's a scalar-shaped tensor.
    sel = {ind: assignment[ind] for ind in t.inds if ind in assignment}
    t_fixed = t.isel(sel) if sel else t

    # After fixing all factor inds, we expect a scalar; get the number from .data
    # If something is left (e.g., you didn't assign all inds), this will not be scalar.
    val = float(np.asarray(t_fixed.data))
    return val


def trace_probability_from_TN_SLOW(local_tensors, Z=None):
    vars_set = set().union(*[set(T.inds) for T in local_tensors])
    vars_order = sorted(vars_set)
    # Enumerate all 2^k assignments to normalize (small neighborhoods only).
    if Z is None:
        Z = 0.0
        for bits in np.ndindex(*(2,) * len(vars_order)):
            a = {v: b for v, b in zip(vars_order, bits)}

            w = 1.0
            # Multiply factor potentials evaluated at the assignment
            for T in local_tensors:
                w *= eval_tensor_at_assignment(T, a)
            Z += w

    #  print("manual calculated Z = ", Z)
    sum = 0
    # calculate prob log prob
    for bits in np.ndindex(*(2,) * len(vars_order)):
        a = {v: b for v, b in zip(vars_order, bits)}
        w = 1.0
        for T in local_tensors:
            w *= eval_tensor_at_assignment(T, a)

        p = w / Z if Z > 0 else 0.0
        sum += p * math.log(p)
    return sum


# efficient implmentation of trace_probability_from_TN_SLOW that uses the fact that we sum over all combinations to its advantage
def trace_probability_from_TN(local_tensors, Z=None):
    vars_set = set().union(*(T.inds for T in local_tensors))
    vars_order = sorted(vars_set)

    # Contract all tensors, leaving vars_order open
    T_all = qu.tensor.tensor_contract(*local_tensors, output_inds=vars_order)

    # Now T_all is a tensor over the variables (shape (2,)*len(vars_order))
    arr = T_all.data

    if Z is None:
        Z = arr.sum()

    # Normalize
    P = arr / Z if Z > 0 else arr

    # Flatten to probability vector
    p_flat = P.reshape(-1)

    # Avoid log(0)
    # mask = p_flat > 0
    sum_prob = np.sum(p_flat * np.log(p_flat))

    # print("SUM using one quimb tensor", float(sum_prob))
    return sum_prob


# efficient implmentation of trace_probability_from_TN_SLOW that uses the fact that we sum over all combinations to its advantage
def trace_probability_from_TN_QUIMB(local_tensors, Z=None):
    if Z is None:
        Z = qtn.TensorNetwork(local_tensors).contract(output_inds=())

    local_network = qtn.TensorNetwork(local_tensors).multiply(1 / Z)

    sum_prob = 0

    for t in (
        tqdm(local_network.tensors)
        if len(local_network.tensors) > 100
        else local_network.tensors
    ):
        t_copy = t.copy()
        local_network_for_t = local_network.copy()

        local_network_for_t.delete(t.tags)

        new_t = qtn.Tensor(
            data=t_copy.data * np.log(t_copy.data), inds=t_copy.inds, tags=t_copy.tags
        )
        local_network_for_t |= new_t
        sum_prob += local_network_for_t.contract(output_inds=(), optimize="auto-hq")

    return sum_prob


def calculate_S_general(
    tn, messages, neighbourhoods_l0, nearest_neighbours, w_tensors, c_tensors
):
    w, c = w_tensors, c_tensors

    S = 0
    print("all neighbour loop tensors")
    for i, j in tqdm(allNeighbours(neighbourhoods_l0)):
        overlap_ij = neighbourhoods_l0[i] & neighbourhoods_l0[j]
        N_i_and_j_tags = [f"N{k}" for k in overlap_ij]

        N_i_and_j_tn = tn.select(N_i_and_j_tags, which="any")

        inner_inds = N_i_and_j_tn.inner_inds()

        S_i_or_j = neighbourhoods_l0[i]
        S_i_tags = [f"N{k}" for k in S_i_or_j]
        S_i__tn = tn.select(S_i_tags, which="any")
        T_i__tn = tn.select([f"N{i}"])
        messages_tn = qtn.TensorNetwork([])

        for k in S_i_or_j - {i}:
            messages_tn |= messages[(k, i)]

        S_i__tn.add(messages_tn)

        big_inner = set(S_i__tn.inner_inds())
        inds_to_contract = set(big_inner) - set(inner_inds)

        for ind in inds_to_contract:
            S_i__tn.contract_ind(ind, output_inds=big_inner - {ind})

        contracted = S_i__tn.contract(output_inds=())
        #  trace_term_test = trace_probability_from_TN([N_i_and_j_tn_test], contracted)
        trace_term_faster = trace_probability_from_TN_QUIMB([S_i__tn], contracted)

        contracted = N_i_and_j_tn.contract(output_inds=())
        #   trace_term = trace_probability_from_TN([N_i_and_j_tn], contracted)
        #  trace_term_faster = trace_probability_from_TN_QUIMB([N_i_and_j_tn], contracted)
        #    print("trace terem faster diff S1", trace_term_faster - trace_term)

        S -= (1 / math.comb(len(overlap_ij), 2)) * trace_term_faster

    for i in range(len(neighbourhoods_l0.items())):
        S_i_or_j = neighbourhoods_l0[i]
        S_i_tags = [f"N{k}" for k in S_i_or_j]
        S_i__tn = tn.select(S_i_tags, which="any")
        T_i__tn = tn.select([f"N{i}"])
        messages_tn = qtn.TensorNetwork([])

        for k in S_i_or_j - {i}:
            messages_tn |= messages[(k, i)]

        S_i__tn.add(messages_tn)

        T = T_i__tn.tensors[0]
        T_tn = S_i__tn.contract(output_inds=(T.inds))
        #    trace_term = trace_probability_from_TN([T_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB([T_tn])
        #  print("trace terem faster diff S2", trace_term_faster - trace_term)

        S -= w[i] * trace_term_faster
    all_nearest_neighbours = allNeighbours(nearest_neighbours)
    for i, j in all_nearest_neighbours:
        S_i_or_j = neighbourhoods_l0[i] | neighbourhoods_l0[j]

        S_i_or_j_tags = [f"N{k}" for k in S_i_or_j]
        S_i_or_j_tn = tn.select(S_i_or_j_tags, which="any")
        messages_tn = qtn.TensorNetwork([])

        for k in S_i_or_j - {i} - {j}:
            if (k, i) not in messages.keys():
                messages_tn |= messages[(k, j)]
                continue
            if (k, j) not in messages.keys():
                messages_tn |= messages[(k, i)]
                continue
            m_to_i = messages[(k, i)]
            m_to_j = messages[(k, j)]
            if len(m_to_i.inds) < len(m_to_j.inds):
                messages_tn |= messages[(k, i)]
            else:
                messages_tn |= messages[(k, j)]

        S_i_or_j.add(messages_tn)

        T_i = tn.select([f"N{i}"]).tensors[0]
        T_j = tn.select([f"N{j}"]).tensors[0]
        shared_ind = [ind for ind in T_i.inds if ind in T_j.inds][0]

        ij_tn = S_i_or_j_tn.contract(output_inds=(shared_ind,))

        #  trace_term = trace_probability_from_TN([ij_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB(([ij_tn]))
        #   print("trace terem faster diff S3", trace_term_faster - trace_term)

        S -= c[i, j] * trace_term_faster

    return S


def calculate_alpha_beta(tn, neighbourhoods_l0, nearest_neighbours):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0.items()}

    NN_tensors = {k: v[0] for k, v in nearest_neighbours.items()}

    allNeighbours_l0 = allNeighbours(N_l0_tensors)
    columns = len(allNeighbours(N_l0_tensors)) + len(tn.tensors)
    rows = len(tn.tensors) + len(allNeighbours(NN_tensors))
    a = np.zeros((rows, columns))
    b = np.ones(rows)

    for row in range(len(tn.tensors)):
        for column in range(len(tn.tensors)):
            if row in N_l0_tensors[column]:
                a[row, column] = 1
        for column in range(
            len(tn.tensors), len(tn.tensors) + len(allNeighbours(N_l0_tensors)), 1
        ):
            i, j = allNeighbours_l0[column - len(tn.tensors)]

            overlap_ij = set(N_l0_tensors[i]) & set(N_l0_tensors[j])
            if row in overlap_ij:
                a[row, column] = 1

    for k, ij in enumerate(allNeighbours(NN_tensors)):
        row = len(tn.tensors) + k
        for column in range(len(tn.tensors)):
            if ij in N_l0_variables[column]:
                a[row, column] = 1
        for column in range(
            len(tn.tensors), len(tn.tensors) + len(allNeighbours(N_l0_tensors)), 1
        ):
            i, j = allNeighbours_l0[column - len(tn.tensors)]
            overlap_ij = set(N_l0_variables[i]) & set(N_l0_variables[j])
            if ij in overlap_ij:
                a[row, column] = 1

    x, _, _, _ = np.linalg.lstsq(a, b)

    return x[: len(tn.tensors)], x[len(tn.tensors) :]


def calculate_S_alpha_beta(tn, messages, neighbourhoods_l0, nearest_neighbours):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0.items()}

    alpha, beta = calculate_alpha_beta(tn, neighbourhoods_l0, nearest_neighbours)

    S = 0
    for i in range(len(N_l0_tensors.items())):
        S_i = N_l0_tensors[i]
        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])
        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        local_tn.add(T_i__tn)
        contracted = trace_probability_from_TN_QUIMB(local_tn.tensors)
        S -= contracted * alpha[i]

    for idx, (i, j) in enumerate(allNeighbours(N_l0_tensors)):
        overlap_ij_variables = N_l0_variables[i] & N_l0_variables[j]
        overlap_ij = set()
        for a, b in overlap_ij_variables:
            overlap_ij.add(a)
            overlap_ij.add(b)

        local_tn = qtn.TensorNetwork()

        for k in overlap_ij - {i} - {j}:
            msg_ki = messages[(k, i)]
            msg_kj = messages[(k, j)]
            msg_tn = qtn.TensorNetwork()

            if not set(msg_ki.inds).issubset(set(msg_kj.inds)):
                msg_tn |= msg_ki
            if not set(msg_kj.inds).issubset(set(msg_ki.inds)):
                msg_tn |= msg_kj
            if set(msg_kj.inds) == set(msg_ki.inds):
                msg_tn |= msg_kj
            # print("messages tn", msg_tn)

            T_k = tn.select(f"N{k}", which="any")
            output_inds = set(T_k.tensors[0].inds)
            for msg_tensor in msg_tn.tensors:
                output_inds -= set(msg_tensor.inds)
            msg_tn = msg_tn.multiply(1 / msg_tn.contract(output_inds=()))
            msg_and_tensor = qtn.TensorNetwork([msg_tn, T_k])
            #    print("msg and tensor", msg_and_tensor)
            combined = msg_and_tensor.contract(output_inds=output_inds)
            #    print("combined ", combined)
            local_tn.add(combined)

        msg_and_tensor = qtn.TensorNetwork(
            [messages[(i, j)], tn.select(f"N{i}", which="any")]
        )
        local_tn.add(msg_and_tensor.contract())
        msg_and_tensor = qtn.TensorNetwork(
            [messages[(j, i)], tn.select(f"N{j}", which="any")]
        )
        local_tn.add(msg_and_tensor.contract())
        #  print("local tn", local_tn)
        contracted = trace_probability_from_TN_QUIMB(local_tn.tensors)
        #    print("contracted", contracted)

        S -= contracted * beta[idx]

    return S


# alpha beta
def calculate_Z_better(
    tn,
    messages,
    neighbourhoods_l0,
    nearest_neighbours,
):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0.items()}

    alpha, beta = calculate_alpha_beta(tn, neighbourhoods_l0, nearest_neighbours)

    Z = 1
    for i in range(len(N_l0_tensors.items())):
        S_i = N_l0_tensors[i]
        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])
        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        local_tn.add(T_i__tn)
        contracted = local_tn.contract()
        Z *= contracted ** alpha[i]

    for idx, (i, j) in enumerate(allNeighbours(N_l0_tensors)):
        overlap_ij_variables = N_l0_variables[i] & N_l0_variables[j]
        overlap_ij = set()
        for a, b in overlap_ij_variables:
            overlap_ij.add(a)
            overlap_ij.add(b)

        local_tn = qtn.TensorNetwork()

        for k in overlap_ij - {i} - {j}:
            msg_ki = messages[(k, i)]
            msg_kj = messages[(k, j)]
            msg_tn = qtn.TensorNetwork()

            if not set(msg_ki.inds).issubset(set(msg_kj.inds)):
                msg_tn |= msg_ki
            if not set(msg_kj.inds).issubset(set(msg_ki.inds)):
                msg_tn |= msg_kj
            if set(msg_kj.inds) == set(msg_ki.inds):
                msg_tn |= msg_kj
            # print("messages tn", msg_tn)

            T_k = tn.select(f"N{k}", which="any")
            output_inds = set(T_k.tensors[0].inds)
            for msg_tensor in msg_tn.tensors:
                output_inds -= set(msg_tensor.inds)
            msg_tn = msg_tn.multiply(1 / msg_tn.contract(output_inds=()))
            msg_and_tensor = qtn.TensorNetwork([msg_tn, T_k])
            #    print("msg and tensor", msg_and_tensor)
            combined = msg_and_tensor.contract(output_inds=output_inds)
            #    print("combined ", combined)
            local_tn.add(combined)

        msg_and_tensor = qtn.TensorNetwork(
            [messages[(i, j)], tn.select(f"N{i}", which="any")]
        )
        local_tn.add(msg_and_tensor.contract())
        msg_and_tensor = qtn.TensorNetwork(
            [messages[(j, i)], tn.select(f"N{j}", which="any")]
        )
        local_tn.add(msg_and_tensor.contract())
        #  print("local tn", local_tn)
        contracted = local_tn.contract()
        #    print("contracted", contracted)

        Z *= contracted ** beta[idx]

    return Z


def calculate_Z_more_better(
    tn,
    messages,
    neighbourhoods_l0,
    nearest_neighbours,
):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0.items()}

    alpha, beta = calculate_alpha_beta(tn, neighbourhoods_l0, nearest_neighbours)

    Z = 1
    for i in range(len(N_l0_tensors.items())):
        S_i = N_l0_tensors[i]
        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])
        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        local_tn.add(T_i__tn)
        contracted = local_tn.contract()
        Z *= contracted ** alpha[i]

    for idx, (i, j) in enumerate(allNeighbours(N_l0_tensors)):
        overlap_ij_variables = N_l0_variables[i] & N_l0_variables[j]
        overlap_ij = set()
        for a, b in overlap_ij_variables:
            overlap_ij.add(a)
            overlap_ij.add(b)

        local_tn = qtn.TensorNetwork()

        for k in overlap_ij - {i} - {j}:
            T_k = tn.select(f"N{k}", which="any")
            msg_ki = messages[(k, i)]
            msg_kj = messages[(k, j)]
            msg_tn = qtn.TensorNetwork()

            if not set(msg_ki.inds).issubset(set(msg_kj.inds)):
                msg_tn |= msg_ki
            if not set(msg_kj.inds).issubset(set(msg_ki.inds)):
                msg_tn |= msg_kj
            if set(msg_kj.inds) == set(msg_ki.inds):
                msg_tn |= msg_kj
            #   print("messages tn", msg_tn)
            if len(msg_tn.tensors) > 1:
                overlap_inds = set(msg_tn.tensors[0].inds)
                for msg_tensor in msg_tn.tensors:
                    overlap_inds &= set(msg_tensor.inds)

                deg = len(overlap_inds)
                shape = (2,) * deg
                data = np.full(shape, 1 / (2**deg), dtype=float)
                data /= np.sum(data)
                uniform_tensor = qtn.Tensor(data=data, inds=overlap_inds)
                msg1_and_uniform = qtn.TensorNetwork(
                    [msg_tn.tensors[0], uniform_tensor]
                )
                contracted_agains_uniform = msg1_and_uniform.contract()

                msg_tn = qtn.TensorNetwork(
                    [msg_tn.tensors[1], contracted_agains_uniform]
                )
            msg_and_tensor = qtn.TensorNetwork([msg_tn, T_k])
            #    print("msg and tensor", msg_and_tensor)
            combined = msg_and_tensor.contract()
            #    print("combined ", combined)
            local_tn.add(combined)

        msg_and_tensor = qtn.TensorNetwork(
            [messages[(i, j)], tn.select(f"N{i}", which="any")]
        )
        local_tn.add(msg_and_tensor.contract())
        msg_and_tensor = qtn.TensorNetwork(
            [messages[(j, i)], tn.select(f"N{j}", which="any")]
        )
        local_tn.add(msg_and_tensor.contract())
        #  print("local tn", local_tn)
        contracted = local_tn.contract()
        # print("contracted", contracted)

        Z *= contracted ** beta[idx]

    return Z


def calculate_log_S(U, Z):
    print(f"Z = {Z} U = {U}")
    if Z > 0:
        return math.log(Z) + U
    print("calculated Z is zero")
    return U


def calculate_S_general_variables(
    tn, messages, neighbourhoods_l0, nearest_neighbours, w, c
):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    NN_tensors = {k: v[0] for k, v in nearest_neighbours.items()}

    S = 0
    for i, j in tqdm(allNeighbours(N_l0_tensors)):
        overlap_ij = N_l0_tensors[i] & N_l0_tensors[j]

        N_i_and_j_tags = [f"N{k}" for k in overlap_ij]

        N_i_and_j_tn = tn.select(N_i_and_j_tags, which="any")

        inner_inds = set(N_i_and_j_tn.inner_inds())

        S_i = N_l0_tensors[i]

        local_tn = qtn.TensorNetwork()

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_ki = messages[(k, i)]
            for n in NN_tensors[k] - {i}:
                if n > k and (n, i) in messages.keys():
                    msg_ni = messages[(n, i)]
                    shared_inds = set(msg_ki.inds) & set(msg_ni.inds)
                    for ind in shared_inds:
                        T_k = T_k.reindex({ind: f"{ind}_{k}"})
                        msg_ki = msg_ki.reindex({ind: f"{ind}_{k}"})
                        if ind in inner_inds:
                            inner_inds.add(f"{ind}_{k}")
            local_tn.add(T_k)
            local_tn.add(msg_ki)

        T_i__tn = tn.select([f"N{i}"])
        local_tn.add(T_i__tn)
        # print("TN ", N_i_and_j_tn)
        # print("inner inds", inner_inds)

        #  N_i_and_j_tn_test = local_tn.contract(output_inds=inner_inds)

        big_inner = set(local_tn.inner_inds())
        inds_to_contract = set(big_inner) - inner_inds

        for ind in inds_to_contract:
            local_tn.contract_ind(ind, output_inds=big_inner - {ind})

        contracted = local_tn.contract(output_inds=())
        #  trace_term_test = trace_probability_from_TN([N_i_and_j_tn_test], contracted)
        trace_term_faster = trace_probability_from_TN_QUIMB([local_tn], contracted)
        #  print("trace terem faster diff S1", trace_term_faster - trace_term_test)

        S -= (1 / math.comb(len(overlap_ij), 2)) * trace_term_faster

    for i in range(len(N_l0_tensors.items())):
        S_i = N_l0_tensors[i]

        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        T = T_i__tn.tensors[0]
        local_tn.add(T_i__tn)
        T_tn = local_tn.contract(output_inds=(T.inds))
        #    trace_term = trace_probability_from_TN([T_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB([T_tn])
        #  print("trace terem faster diff S2", trace_term_faster - trace_term)

        S -= w[i] * trace_term_faster

    all_nearest_neighbours = allNeighbours(NN_tensors)
    for i, j in all_nearest_neighbours:
        S_i_or_j = N_l0_tensors[i] | N_l0_tensors[j]

        local_tn = qtn.TensorNetwork([])

        messages_and_tensors_to_add = set()

        for k in S_i_or_j - {i} - {j}:
            if (k, i) not in messages.keys() and (k, j) not in messages.keys():
                local_tn.add(tn.select(f"N{k}", which="any"))
                continue
            if (k, i) not in messages.keys():
                messages_and_tensors_to_add.add(((k, j), k))
                continue
            if (k, j) not in messages.keys():
                messages_and_tensors_to_add.add(((k, i), k))
                continue
            m_to_i = messages[(k, i)]
            m_to_j = messages[(k, j)]
            if len(m_to_i.inds) < len(m_to_j.inds):
                messages_and_tensors_to_add.add(((k, i), k))
            else:
                messages_and_tensors_to_add.add(((k, j), k))

        for msg, tensor in messages_and_tensors_to_add:
            T_k = tn.select(f"N{tensor}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[msg], T_k])
            local_tn.add(msg_and_tensor.contract())

        T_i = tn.select([f"N{i}"]).tensors[0]
        T_j = tn.select([f"N{j}"]).tensors[0]
        local_tn.add(T_i)
        local_tn.add(T_j)
        shared_ind = [ind for ind in T_i.inds if ind in T_j.inds][0]
        # print("TN ", local_tn)
        # print("inner inds", shared_ind)
        ij_tn = local_tn.contract(output_inds=(shared_ind,))

        #  trace_term = trace_probability_from_TN([ij_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB(([ij_tn]))
        #   print("trace terem faster diff S3", trace_term_faster - trace_term)

        S -= c[i, j] * trace_term_faster

    return S


def calculate_S_general_variables_better(
    tn, messages, neighbourhoods_l0, nearest_neighbours, w, c
):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0.items()}
    NN_tensors = {k: v[0] for k, v in nearest_neighbours.items()}

    S = 0
    for i, j in tqdm(allNeighbours(N_l0_tensors)):
        overlap_ij_variables = N_l0_variables[i] & N_l0_variables[j]
        overlap_ij = set()
        for a, b in overlap_ij_variables:
            overlap_ij.add(a)
            overlap_ij.add(b)

        N_i_and_j_tags = [f"N{k}" for k in overlap_ij]

        N_i_and_j_tn = tn.select(N_i_and_j_tags, which="any")

        inner_inds = set(N_i_and_j_tn.inner_inds())

        S_i = N_l0_tensors[i]

        local_tn = qtn.TensorNetwork()

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_ki = messages[(k, i)]
            for n in NN_tensors[k] - {i}:
                if n > k and (n, i) in messages.keys():
                    msg_ni = messages[(n, i)]
                    shared_inds = set(msg_ki.inds) & set(msg_ni.inds)
                    for ind in shared_inds:
                        T_k = T_k.reindex({ind: f"{ind}_{k}"})
                        msg_ki = msg_ki.reindex({ind: f"{ind}_{k}"})
                        if ind in inner_inds:
                            inner_inds.add(f"{ind}_{k}")
            local_tn.add(T_k)
            local_tn.add(msg_ki)

        T_i__tn = tn.select([f"N{i}"])
        local_tn.add(T_i__tn)
        # print("TN ", N_i_and_j_tn)
        # print("inner inds", inner_inds)

        #  N_i_and_j_tn_test = local_tn.contract(output_inds=inner_inds)

        big_inner = set(local_tn.inner_inds())
        inds_to_contract = set(big_inner) - inner_inds

        for ind in inds_to_contract:
            local_tn.contract_ind(ind, output_inds=big_inner - {ind})

        contracted = local_tn.contract(output_inds=())
        #  trace_term_test = trace_probability_from_TN([N_i_and_j_tn_test], contracted)
        trace_term_faster = trace_probability_from_TN_QUIMB([local_tn], contracted)
        #  print("trace terem faster diff S1", trace_term_faster - trace_term_test)

        S -= (1 / math.comb(len(overlap_ij), 2)) * trace_term_faster

    for i in range(len(N_l0_tensors.items())):
        S_i = N_l0_tensors[i]

        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        T = T_i__tn.tensors[0]
        local_tn.add(T_i__tn)
        T_tn = local_tn.contract(output_inds=(T.inds))
        #    trace_term = trace_probability_from_TN([T_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB([T_tn])
        #  print("trace terem faster diff S2", trace_term_faster - trace_term)

        S -= w[i] * trace_term_faster

    all_nearest_neighbours = allNeighbours(NN_tensors)
    for i, j in all_nearest_neighbours:
        S_i_or_j = N_l0_tensors[i] | N_l0_tensors[j]

        local_tn = qtn.TensorNetwork([])

        messages_and_tensors_to_add = set()

        for k in S_i_or_j - {i} - {j}:
            if (k, i) not in messages.keys() and (k, j) not in messages.keys():
                local_tn.add(tn.select(f"N{k}", which="any"))
                continue
            if (k, i) not in messages.keys():
                messages_and_tensors_to_add.add(((k, j), k))
                continue
            if (k, j) not in messages.keys():
                messages_and_tensors_to_add.add(((k, i), k))
                continue
            m_to_i = messages[(k, i)]
            m_to_j = messages[(k, j)]
            if len(m_to_i.inds) < len(m_to_j.inds):
                messages_and_tensors_to_add.add(((k, i), k))
            else:
                messages_and_tensors_to_add.add(((k, j), k))

        for msg, tensor in messages_and_tensors_to_add:
            T_k = tn.select(f"N{tensor}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[msg], T_k])
            local_tn.add(msg_and_tensor.contract())

        T_i = tn.select([f"N{i}"]).tensors[0]
        T_j = tn.select([f"N{j}"]).tensors[0]
        local_tn.add(T_i)
        local_tn.add(T_j)
        shared_ind = [ind for ind in T_i.inds if ind in T_j.inds][0]
        # print("TN ", local_tn)
        # print("inner inds", shared_ind)
        ij_tn = local_tn.contract(output_inds=(shared_ind,))

        #  trace_term = trace_probability_from_TN([ij_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB(([ij_tn]))
        #   print("trace terem faster diff S3", trace_term_faster - trace_term)

        S -= c[i, j] * trace_term_faster

    return S


def calculate_Z_general_variables(tn, messages, neighbourhoods_l0, nearest_neighbours):
    N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
    N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0.items()}

    NN_tensors = {k: v[0] for k, v in nearest_neighbours.items()}
    NN_variables = {k: v[1] for k, v in nearest_neighbours.items()}

    w, c = factors_general_variables(
        N_l0_tensors=N_l0_tensors,
        N_l0_variables=N_l0_variables,
        NN_tensors=NN_tensors,
        NN_variables=NN_variables,
    )

    Z = 1
    for i, j in tqdm(allNeighbours(N_l0_tensors)):
        overlap_ij = N_l0_tensors[i] & N_l0_tensors[j]

        N_i_and_j_tags = [f"N{k}" for k in overlap_ij]

        N_i_and_j_tn = tn.select(N_i_and_j_tags, which="any")

        inner_inds = set(N_i_and_j_tn.inner_inds())

        S_i = N_l0_tensors[i]

        local_tn = qtn.TensorNetwork()

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_ki = messages[(k, i)]
            for n in NN_tensors[k] - {i}:
                if n > k and (n, i) in messages.keys():
                    msg_ni = messages[(n, i)]
                    shared_inds = set(msg_ki.inds) & set(msg_ni.inds)
                    for ind in shared_inds:
                        T_k = T_k.reindex({ind: f"{ind}_{k}"})
                        msg_ki = msg_ki.reindex({ind: f"{ind}_{k}"})
                        if ind in inner_inds:
                            inner_inds.add(f"{ind}_{k}")
            local_tn.add(T_k)
            local_tn.add(msg_ki)

        T_i__tn = tn.select([f"N{i}"])
        local_tn.add(T_i__tn)
        # print("TN ", N_i_and_j_tn)
        # print("inner inds", inner_inds)
        N_i_and_j_tn = local_tn.contract(output_inds=inner_inds)

        contracted = N_i_and_j_tn.contract(output_inds=())
        #   trace_term = trace_probability_from_TN([N_i_and_j_tn], contracted)
        #    print("trace terem faster diff S1", trace_term_faster - trace_term)

        Z *= contracted ** (1 / math.comb(len(overlap_ij), 2))

    for i in range(len(N_l0_tensors.items())):
        S_i = N_l0_tensors[i]

        T_i__tn = tn.select([f"N{i}"])
        local_tn = qtn.TensorNetwork([])

        for k in S_i - {i}:
            T_k = tn.select(f"N{k}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
            local_tn.add(msg_and_tensor.contract())

        T = T_i__tn.tensors[0]
        local_tn.add(T_i__tn)
        contracted = local_tn.contract(output_inds=())
        #    trace_term = trace_probability_from_TN([T_tn])

        #  print("trace terem faster diff S2", trace_term_faster - trace_term)

        Z *= contracted ** w[i]

    all_nearest_neighbours = allNeighbours(NN_tensors)
    for i, j in all_nearest_neighbours:
        S_i_or_j = N_l0_tensors[i] | N_l0_tensors[j]

        local_tn = qtn.TensorNetwork([])

        messages_and_tensors_to_add = set()

        for k in S_i_or_j - {i} - {j}:
            if (k, i) not in messages.keys() and (k, j) not in messages.keys():
                local_tn.add(tn.select(f"N{k}", which="any"))
                continue
            if (k, i) not in messages.keys():
                messages_and_tensors_to_add.add(((k, j), k))
                continue
            if (k, j) not in messages.keys():
                messages_and_tensors_to_add.add(((k, i), k))
                continue
            m_to_i = messages[(k, i)]
            m_to_j = messages[(k, j)]
            if len(m_to_i.inds) < len(m_to_j.inds):
                messages_and_tensors_to_add.add(((k, i), k))
            else:
                messages_and_tensors_to_add.add(((k, j), k))

        for msg, tensor in messages_and_tensors_to_add:
            T_k = tn.select(f"N{tensor}", which="any")
            msg_and_tensor = qtn.TensorNetwork([messages[msg], T_k])
            local_tn.add(msg_and_tensor.contract())

        T_i = tn.select([f"N{i}"]).tensors[0]
        T_j = tn.select([f"N{j}"]).tensors[0]
        local_tn.add(T_i)
        local_tn.add(T_j)
        # print("TN ", local_tn)
        # print("inner inds", shared_ind)
        contracted = local_tn.contract(output_inds=())

        #  trace_term = trace_probability_from_TN([ij_tn])
        # trace_term_faster = trace_probability_from_TN_QUIMB(([ij_tn]))
        #   print("trace terem faster diff S3", trace_term_faster - trace_term)

        Z *= contracted ** c[i, j]

    return Z


def calculate_S_general_cheap_intersection(
    tn, messages, neighbourhoods_l0, nearest_neighbours
):
    w, c = factors_general(neighbourhoods_l0, nearest_neighbours)

    S = 0
    for i, j in tqdm(allNeighbours(neighbourhoods_l0)):
        overlap_ij = neighbourhoods_l0[i] & neighbourhoods_l0[j]
        N_i_and_j_tags = [f"N{k}" for k in overlap_ij]

        N_i_and_j_tn = tn.select(N_i_and_j_tags, which="any")

        #  inner_inds = N_i_and_j_tn.inner_inds()
        messages_tn = qtn.TensorNetwork([])
        messages_tn |= messages[(i, j)]
        messages_tn |= messages[(j, i)]

        for k in overlap_ij - {i} - {j}:
            mki = messages[(k, i)]
            mkj = messages[(k, j)]
            if not set(mki.inds).issubset(set(mkj.inds)):
                messages_tn |= mki
            if not set(mkj.inds).issubset(set(mki.inds)):
                messages_tn |= mkj

        N_i_and_j_tn.add(messages_tn)

        contracted = N_i_and_j_tn.contract()
        print("contracted", contracted)
        #   trace_term = trace_probability_from_TN([N_i_and_j_tn], contracted)
        trace_term_faster = trace_probability_from_TN_QUIMB([N_i_and_j_tn], contracted)
        #    print("trace terem faster diff S1", trace_term_faster - trace_term)

        S -= (1 / math.comb(len(overlap_ij), 2)) * trace_term_faster

    for i in range(len(neighbourhoods_l0.items())):
        S_i_or_j = neighbourhoods_l0[i]
        S_i_tags = [f"N{k}" for k in S_i_or_j]
        S_i__tn = tn.select(S_i_tags, which="any")
        T_i__tn = tn.select([f"N{i}"])
        messages_tn = qtn.TensorNetwork([])

        for k in S_i_or_j - {i}:
            messages_tn |= messages[(k, i)]

        S_i__tn.add(messages_tn)

        T = T_i__tn.tensors[0]
        T_tn = S_i__tn.contract(output_inds=(T.inds))
        #    trace_term = trace_probability_from_TN([T_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB([T_tn])
        #  print("trace terem faster diff S2", trace_term_faster - trace_term)

        S -= w[i] * trace_term_faster
    all_nearest_neighbours = allNeighbours(nearest_neighbours)
    for i, j in all_nearest_neighbours:
        S_i_or_j = neighbourhoods_l0[i] | neighbourhoods_l0[j]

        S_i_or_j_tags = [f"N{k}" for k in S_i_or_j]
        S_i_or_j_tn = tn.select(S_i_or_j_tags, which="any")
        messages_tn = qtn.TensorNetwork([])

        for k in S_i_or_j - {i} - {j}:
            if (k, i) not in messages.keys():
                messages_tn |= messages[(k, j)]
                continue
            if (k, j) not in messages.keys():
                messages_tn |= messages[(k, i)]
                continue
            m_to_i = messages[(k, i)]
            m_to_j = messages[(k, j)]
            if len(m_to_i.inds) < len(m_to_j.inds):
                messages_tn |= messages[(k, i)]
            else:
                messages_tn |= messages[(k, j)]

        S_i_or_j.add(messages_tn)

        T_i = tn.select([f"N{i}"]).tensors[0]
        T_j = tn.select([f"N{j}"]).tensors[0]
        shared_ind = [ind for ind in T_i.inds if ind in T_j.inds][0]

        ij_tn = S_i_or_j_tn.contract(output_inds=(shared_ind,))

        #  trace_term = trace_probability_from_TN([ij_tn])
        trace_term_faster = trace_probability_from_TN_QUIMB(([ij_tn]))
        #   print("trace terem faster diff S3", trace_term_faster - trace_term)

        S -= c[i, j] * trace_term_faster

    return S


def calculate_S_exact_test(tn):
    for i in range(len(tn.tensors)):
        S_quimb = -trace_probability_from_TN_QUIMB(tn.tensors[0 : i + 1])
        S_big_tensor = -trace_probability_from_TN(tn.tensors[0 : i + 1])
        S_slow = -trace_probability_from_TN_SLOW(tn.tensors[0 : i + 1])
        print(f"{i + 1} tensors")
        print("quimb: ", S_quimb)
        print("big t: ", S_big_tensor)
        print("slow : ", S_slow)
        print("--------------")


def calculate_S_exact(tn):
    return -trace_probability_from_TN_QUIMB(tn.tensors)


def partition_function_SU(S, U):
    return math.exp((S - U))


def partition_function_SUbeta(S, U, beta):
    return math.exp((S - beta * U))
