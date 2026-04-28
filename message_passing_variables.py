import numpy as np
import quimb.tensor as qtn
from tqdm import tqdm


def dfs_variables(path, visited, max_length, n, adj_matrix, start, nodes, variables):
    if len(path) > max_length:
        return
    node = path[-1]
    for neighbor in range(n):
        if adj_matrix[node][neighbor]:
            if neighbor == start and len(path) > 2:
                # Found a cycle
                nodes |= set(path)
                for i in range(len(path) - 1):
                    variables.add(
                        (path[i], path[i + 1])
                        if path[i] < path[i + 1]
                        else (path[i + 1], path[i])
                    )
            elif neighbor not in visited:
                dfs_variables(
                    path=path + [neighbor],
                    visited=visited | {neighbor},
                    max_length=max_length,
                    n=n,
                    adj_matrix=adj_matrix,
                    nodes=nodes,
                    variables=variables,
                    start=start,
                )


def find_cycles_through_node_variables(adj_matrix, start, max_length):
    n = len(adj_matrix)

    nodes = set()
    variables = set()
    dfs_variables(
        path=[start],
        visited={start},
        max_length=max_length,
        n=n,
        adj_matrix=adj_matrix,
        nodes=nodes,
        variables=variables,
        start=start,
    )
    return nodes, variables


def get_general_neighborhood_variables(i, l0, adj):
    neighbourhood_tensors = {i}
    neighbourhood_variables = set({})

    # (1) All nodes directly connected
    for j in range(adj.shape[0]):
        if adj[i, j] != 0:
            neighbourhood_tensors.add(j)
            ordered = (i, j) if i < j else (j, i)
            neighbourhood_variables.add(ordered)

    # (2) All nodes in simple cycles containing i with length < l0

    tensors, variables = find_cycles_through_node_variables(adj, i, l0 + 2)
    # if l0 == 0:
    #     print(f"0 cycle nodes in neighborhood for {i}: ", neighbourhood)

    return (tensors | neighbourhood_tensors, variables | neighbourhood_variables)


# general tns grouped by variables
def message_passing_general_variables(
    tn,
    neighbourhoods_l0,
    neighbourhoods_nearest,
    num_iterations=20,
    eps=1e-12,
    delta_new=1,
):
    messages, messages_intersection = initialize_messages_general_variables(
        neighbourhoods_l0, neighbourhoods_nearest
    )

    #   print("messages", messages)
    oldmessages, old_messages_intersection = initialize_messages_general_variables(
        neighbourhoods_l0, neighbourhoods_nearest
    )

    new_messages = {}
    new_messages_intersection = {}
    for t in tqdm(range(num_iterations)):
        #  print("----------------------------   ITERATION ", t)
        diff = 0
        for i in range(len(tn.tensors)):
            N_i, _ = neighbourhoods_l0[i]
            for n in N_i - {i}:
                new_messages[(i, n)] = update_message_general_variables(
                    i=i,
                    j=n,
                    messages=messages,
                    intersection_messages=messages_intersection,
                    tn=tn,
                    neighbourhoods_l0=neighbourhoods_l0,
                    delta_new=delta_new,
                )
                if t > 0:
                    difference_tensor = new_messages[(i, n)] - oldmessages[(i, n)]
                    #  print("differnence", (difference_tensor.data))
                    diff += np.sum(abs(difference_tensor.data))
                tuplein = (i, n) if i < n else (n, i)
                new_messages_intersection[tuplein, i] = (
                    update_message_general_intersection_variables(
                        i=i,
                        j=n,
                        messages=messages,
                        intersection_messages=messages_intersection,
                        tn=tn,
                        neighbourhoods_l0=neighbourhoods_l0,
                        nearest_neighbourhoods=neighbourhoods_nearest,
                        delta_new=delta_new,
                    )
                )
                if t > 0:
                    difference_tensor = (
                        new_messages_intersection[tuplein, i]
                        - old_messages_intersection[tuplein, i]
                    )
                    diff += np.sum(abs(difference_tensor.data))

        # some messages don't get updated because they are on the edge
        for k, v in messages.items():
            if k not in new_messages:
                print("message not updated", k, v)
                new_messages[k] = v
        messages = new_messages.copy()
        messages_intersection = new_messages_intersection.copy()
        oldmessages = new_messages.copy()
        old_messages_intersection = new_messages_intersection.copy()
        print("diff: ", diff)
        if t > 1 and diff <= eps:
            print(f"converged after {t} iterations")
            break
    return messages.copy()


# messages from i to j
def update_message_general_variables(
    i, j, messages, intersection_messages, tn, neighbourhoods_l0, delta_new=1
):
    if len(messages[(i, j)].inds) == 0:
        return messages[(i, j)]
    N_i_tensors, N_i_variables = neighbourhoods_l0[i]
    N_j_tensors, N_j_variables = neighbourhoods_l0[j]
    neighbordiff_variables = set(N_i_variables) - set(N_j_variables)
    neighbordiff_tensors = set()
    for a, b in neighbordiff_variables:
        neighbordiff_tensors.add(a)
        neighbordiff_tensors.add(b)
    local_tn = qtn.TensorNetwork([])

    for k in neighbordiff_tensors - {i}:
        T_k = tn.select(f"N{k}", which="any")
        msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
        local_tn.add(msg_and_tensor.contract())

    tuple = (j, i) if j < i else (i, j)

    local_tn |= intersection_messages[tuple, i]

    if len(local_tn.tensors) != 0:
        try:
            message_new = local_tn.contract()
            if isinstance(message_new, float):
                # print("float")
                # print(f"from {i} to {j}")
                return messages[i, j]
            if set(message_new.inds) != set(messages[(i, j)].inds):
                print("-------------------ERROR--------------")
                print("TN", local_tn)
                print("changed inds from ", messages[(i, j)])
                print("To ", message_new)
                print("---------------------------------")
                assert False
        except:
            print(f"error updating m from {i} to {j}")
            print("local tn", local_tn)
            assert False
    else:
        assert False

    normalization = np.sum(message_new.data)

    if normalization != 0:
        message_new.modify(data=message_new.data / normalization)

    # message_new.modify(
    #     data=delta_new * message_new.data + (1 - delta_new) * messages[(i, j)].data
    # )
    message_new.modify(tags={f"M_{i}_to_{j}"})
    # print("new msg", message_new.data)
    # print("old msg", messages[(i, j)].data)

    return message_new


def update_message_general_intersection_variables(
    i,
    j,
    messages,
    intersection_messages,
    tn,
    neighbourhoods_l0,
    nearest_neighbourhoods,
    delta_new=1,
):
    tupleij = (j, i) if j < i else (i, j)
    if len(messages[i, j].inds) == 0:
        return intersection_messages[tupleij, i]

    N_i_tensors, N_i_variables = neighbourhoods_l0[i]
    N_j_tensors, N_j_variables = neighbourhoods_l0[j]

    intersection_tensors = N_i_tensors & N_j_tensors
    N_i_minus_j_variables = set(N_i_variables) - set(N_j_variables)

    for tensor in intersection_tensors.copy() - {i}:
        _, variables = nearest_neighbourhoods[tensor]
        for variable in variables:
            if variable in N_i_minus_j_variables:
                intersection_tensors -= {tensor}

    local_tn = qtn.TensorNetwork([])

    for k in set(intersection_tensors) - {i}:
        T_k = tn.select(f"N{k}", which="any")
        msg_and_tensor = qtn.TensorNetwork([messages[(k, i)], T_k])
        local_tn.add(msg_and_tensor.contract())

    T_i = tn.select(f"N{i}", which="any")
    msg_and_tensor = qtn.TensorNetwork([messages[(i, j)], T_i])
    local_tn.add(msg_and_tensor.contract())

    if len(local_tn.tensors) != 0:
        message_new = local_tn.contract()
        if isinstance(message_new, float):
            return intersection_messages[tupleij, i]
        if set(message_new.inds) != set(intersection_messages[tupleij, i].inds):
            print("------------------------------ERROR___________")
            print("local_tn intersection", local_tn)
            print("changed inds from ", intersection_messages[tupleij, i])
            print("To ", message_new)
            print("--------------------------------")
            assert False
    else:
        assert False

    normalization = np.sum(message_new.data)

    if normalization != 0:
        message_new.modify(data=message_new.data / normalization)

    if i < j:
        message_new.modify(tags={f"M_{i}_and{j}_to_{i}"})
    else:
        message_new.modify(tags={f"M_{j}_and{i}_to_{i}"})

    return message_new


def initialize_messages_general_variables(neighbourhoods_l0, neighbourhoods_nearest):
    messages = {}
    messages_intersection = {}
    inds = {}

    for i, (N_i_tensors, N_i_variables) in neighbourhoods_l0.items():
        for j in N_i_tensors - {i}:
            inds[(j, i)] = []
            _, NN_j_variables = neighbourhoods_nearest[j]
            for k in NN_j_variables:
                if k not in N_i_variables:
                    inds[(j, i)].append(k)
            inds_strings = [f"bond_{k}_{m}" for k, m in inds[(j, i)]]
            deg = len(inds[(j, i)])
            shape = (2,) * deg
            data = np.full(shape, 1 / (2**deg), dtype=float)
            data /= np.sum(data)

            messages[(j, i)] = qtn.Tensor(
                data=data,
                inds=inds_strings,
                tags={f"M_{j}_to_{i}"},
            )

    for i, (N_i_tensors, N_i_variables) in neighbourhoods_l0.items():
        for j in N_i_tensors - {i}:
            tuple = (j, i) if j < i else (i, j)

            if len(messages[i, j].inds) == 0:
                messages_intersection[tuple, i] = qtn.Tensor(
                    data=1,
                    inds=(),
                    tags={f"M_{j}_and_{i}_to_{i}"},
                )
                continue
            # intersection
            inds_intersection = set()
            N_j_tensors, N_j_variables = neighbourhoods_l0[j]
            set_N_i_minus_set_N_j_variables = N_i_variables - N_j_variables

            tensors_in_N_i_minus_set_N_j = set()
            for a, b in set_N_i_minus_set_N_j_variables:
                if a != i and b != i:
                    tensors_in_N_i_minus_set_N_j.add(a)
                    tensors_in_N_i_minus_set_N_j.add(b)

            contained_vars = set()
            for t1 in tensors_in_N_i_minus_set_N_j:
                for t2 in tensors_in_N_i_minus_set_N_j:
                    ordered = (t1, t2) if t1 < t2 else (t2, t1)
                    contained_vars.add(ordered)

            for d in (N_i_tensors & N_j_tensors) - {i}:
                _, variables_d = neighbourhoods_nearest[d]
                if len(variables_d & set_N_i_minus_set_N_j_variables) > 0:
                    for variable in variables_d:
                        if (
                            (variable not in set_N_i_minus_set_N_j_variables)
                            and variable not in inds[(d, i)]
                            and variable not in contained_vars
                        ):
                            a, b = variable
                            ordered = (a, b) if a < b else (b, a)
                            inds_intersection.add(ordered)

            inds_strings_intersection = [
                f"bond_{d}_{k}" if d < k else f"bond_{k}_{d}"
                for d, k in inds_intersection
            ]
            deg = len(inds_intersection)
            shape = (2,) * deg
            data = np.full(shape, 1 / (2**deg), dtype=float)
            messages_intersection[tuple, i] = qtn.Tensor(
                data=data,
                inds=inds_strings_intersection,
                tags={f"M_{j}_and_{i}_to_{i}"},
            )

    return messages, messages_intersection
