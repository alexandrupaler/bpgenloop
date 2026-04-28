from datetime import datetime
import time
import numpy as np
from tqdm import tqdm
from compute_values import (
    calculate_S_exact,
    calculate_Z_better,
    calculate_l0_neighbourhoods_variables,
    calculate_log_S,
    compute_energy_exact_new,
    compute_energy_general_variables,
    compute_marginals_exact_general,
    compute_marginals_general_variables,
    factors_general_variables,
)
from network_creating import (
    build_tn_from_adj,
)
from message_passing_variables import message_passing_general_variables

import pickle, os, hashlib, numpy as np

CACHE_FILE = "sources/exact_cache.pkl"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def make_key(adj, T):
    # Stable hash from adjacency matrix + T
    adj_bytes = adj.astype(np.int8).tobytes()
    h = hashlib.sha256(adj_bytes).hexdigest()  # matrix hash
    return f"{h}_T{T:.6f}"


def compute_difference_general(approx, exact, percent=True):
    avg_distance = 0
    for i in approx.keys():
        if percent:
            difference_tensor = (approx[i] - exact[i]) / exact[i]
        else:
            difference_tensor = approx[i] - exact[i]

        local_distance = np.sum(abs(difference_tensor.data))

        avg_distance += local_distance / difference_tensor.data.size
    avg_distance /= len(approx.keys())
    return avg_distance


def times_graph_precalc(times):
    latex = []
    latex.append(f"% --- Plot for precalculation times ---")
    latex.append(r"\begin{figure}[h]")
    latex.append(r"\centering")
    latex.append(r"\begin{tikzpicture}")
    latex.append(r"\begin{axis}[")
    latex.append(r"    xlabel={$l_0$},")
    latex.append(r"    ylabel={runtime},")
    latex.append(r"    ymode=log,")  # log scale for errors
    latex.append(r"    legend style={at={(1.05,1)},anchor=north west},")
    latex.append(r"    width=0.8\textwidth, height=0.5\textwidth")
    latex.append(r"]")

    coords = []
    for l0, val in times.items():
        coords.append(f"({l0},{val})")
    latex.append(r"\addplot coordinates {" + " ".join(coords) + "};")
    latex.append(r"\addlegendentry{variables}")

    latex.append(r"\end{axis}")
    latex.append(r"\end{tikzpicture}")
    latex.append(r"\caption{avg runtime relative to smallest l0 value}")
    latex.append(r"\end{figure}")
    latex.append("")

    return "\n".join(latex)


def times_graph(avg_times):
    latex = []
    latex.append(f"% --- Plot for times ---")
    latex.append(r"\begin{figure}[h]")
    latex.append(r"\centering")
    latex.append(r"\begin{tikzpicture}")
    latex.append(r"\begin{axis}[")
    latex.append(r"    xlabel={$l_0$},")
    latex.append(r"    ylabel={runtime},")
    latex.append(r"    ymode=log,")  # log scale for errors
    latex.append(r"    legend style={at={(1.05,1)},anchor=north west},")
    latex.append(r"    width=0.8\textwidth, height=0.5\textwidth")
    latex.append(r"]")

    coords = []

    for l0, val in enumerate(avg_times):
        coords.append(f"({l0},{val})")
    latex.append(r"\addplot coordinates {" + " ".join(coords) + "};")
    latex.append(r"\addlegendentry{variables}")

    latex.append(r"\end{axis}")
    latex.append(r"\end{tikzpicture}")
    latex.append(r"\caption{avg runtime relative to bp}")
    latex.append(r"\end{figure}")
    latex.append("")

    return "\n".join(latex)


def results_to_latex(results, observables=("Z", "S", "U", "marginals")):
    """
    Convert results dict into LaTeX pgfplots code for graphs (new structure).

    Now expects: results[observable][T]["variables_l0={l0}"]

    Parameters
    ----------
    results : dict
        Dictionary with structure results[observable][T][key].
    observables : tuple
        Which observables to generate plots for.

    Returns
    -------
    str
        LaTeX string with one TikZ/pgfplots figure per observable.
    """
    latex = []
    for obs in observables:
        if obs not in results:
            continue
        latex.append(f"% --- Plot for {obs} ---")
        latex.append(r"\begin{figure}[h]")
        latex.append(r"\centering")
        latex.append(r"\begin{tikzpicture}")
        latex.append(r"\begin{axis}[")
        latex.append(r"    xlabel={$T$},")
        latex.append(r"    ylabel={Relative error},")
        latex.append(r"    ymode=log,")  # log scale for errors
        latex.append(r"    legend style={at={(1.05,1)},anchor=north west},")
        latex.append(r"    width=0.8\textwidth, height=0.5\textwidth")
        latex.append(r"]")

        # Loop over all method_l0 keys
        for method_l0, T_dict in results[obs].items():
            # Build coordinates (T, error)
            coords = []
            for T, val in sorted(T_dict.items(), key=lambda x: x[0]):
                coords.append(f"({T},{val})")
            if coords:
                latex.append(r"\addplot coordinates {" + " ".join(coords) + "};")
                latex.append(r"\addlegendentry{" + f"{method_l0}" + r"}")

        latex.append(r"\end{axis}")
        latex.append(r"\end{tikzpicture}")
        latex.append(r"\caption{Relative error for " + obs + "}")
        latex.append(r"\end{figure}")
        latex.append("")

    return "\n".join(latex)


""" Main testing function that generates plots
    adj: adjacency matrix of the graph
    filename: name of the output file (without .tex)
    l_0_values: list of l_0 values to test
    T_values: list of temperature values to test
    quantities: list of quantities to compute errors/plots for
    use_precomputed: whether to use precomputed exact values from cache
    normalized_tn: whether to normalize the tensor network to have Z=1 (this helps with numerical stability on larger graphs)
    printing: whether to print progress messages
"""


def test_plots(
    adj,
    filename,
    l_0_values=[0, 1, 2, 3, 4, 5, 6],
    T_values=[(x + 1) * 0.25 for x in range(16)],
    quantities=[ "Z", "U", "S", "marginals"],
    use_precomputed=True,
    normalized_tn=True,
    printing=False,
    only_runtime=False,
):
    cache = load_cache()

    methods = [""]

    results = {}
    for quantity in quantities:
        results[quantity] = {}
        for method in methods:
            for l_0 in l_0_values:
                results[quantity][f"{method}l0 = {l_0}"] = {}

    times_variables = np.zeros((len(T_values), len(l_0_values)))
    times_precalc = {}

    neighbourhoods_l0_dict = {}
    factors_dict = {}
    if printing:
        print("precomputing nearest neighbourhoods")
    nearest_neighbourhoods = calculate_l0_neighbourhoods_variables(adj=adj, l0=0)
    # precalculating   neighbourhoods and w,c
    if printing:
        print("precomputing l0 NH and w,c")
    for l0 in l_0_values:
        if printing:
            print("l_0 = ", l0)
        # variables method
        start = time.perf_counter()
        neighbourhoods_l0_dict[l0] = calculate_l0_neighbourhoods_variables(
            adj=adj, l0=l0
        )
        N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0_dict[l0].items()}
        N_l0_variables = {k: v[1] for k, v in neighbourhoods_l0_dict[l0].items()}
        NN_tensors = {k: v[0] for k, v in nearest_neighbourhoods.items()}
        NN_variables = {k: v[1] for k, v in nearest_neighbourhoods.items()}

        factors_dict[l0] = factors_general_variables(
            N_l0_tensors=N_l0_tensors,
            N_l0_variables=N_l0_variables,
            NN_tensors=NN_tensors,
            NN_variables=NN_variables,
        )
        end = time.perf_counter()
        times_precalc[l0] = end - start

    for i, T in enumerate(T_values):
        b = 1 / T
        tn = build_tn_from_adj(adj, beta=b, normalized=normalized_tn)
        if printing:
            print("original Z value: ", tn.contract())
        # for faster repeated calculations we use a chache to store exact values
        key = make_key(adj, T)
        if key in cache and use_precomputed:
            if printing:
                print("Loading exact values from cache...")
            Z, S_exact, U_exact, marginals_exact = cache[key]
        else:
            if printing:
                print("calculating exact values")
            Z = tn.contract()
            if printing:
                print("partition function", Z)
                print("calculating S_exact")
            S_exact = calculate_S_exact(tn)
            if printing:
                print("exact S computed")
            U_exact = compute_energy_exact_new(tn, Z)
            if printing:
                print(f"exact U calculated {U_exact}")
            marginals_exact = compute_marginals_exact_general(tn, Z)
            if printing:
                print("computed marginals")
            cache[key] = (Z, S_exact, U_exact, marginals_exact)
            save_cache(cache)
            if printing:
                print("saved cache")

        iterations = 20
        for j, l0 in enumerate(l_0_values):
            start = time.perf_counter()
            neighbourhoods_l0 = neighbourhoods_l0_dict[l0]
            N_l0_tensors = {k: v[0] for k, v in neighbourhoods_l0.items()}
            if printing:
                print("calculating messages")
            messages = message_passing_general_variables(
                tn=tn,
                neighbourhoods_l0=neighbourhoods_l0,
                neighbourhoods_nearest=nearest_neighbourhoods,
                num_iterations=iterations,
                eps=1e-12,
                delta_new=1,
            )
            if "U" in quantities or "S" in quantities:
                if printing:
                    print("calculating U approx")

                U_approx = compute_energy_general_variables(
                    tn=tn, messages=messages, neighbourhoods_l0=neighbourhoods_l0
                )
            else:
                U_approx = 0

            if "marginals" in quantities:
                if printing:
                    print("calculating marginals approx")
                marginals_approx = compute_marginals_general_variables(
                    tn=tn, messages=messages, N_l0_tensors=N_l0_tensors
                )
            else:
                marginals_approx = {}

            if "Z" in quantities or "S" in quantities:
                if printing:
                    print("calculating Z approx")
                Z_better = calculate_Z_better(
                    tn, messages, neighbourhoods_l0, nearest_neighbourhoods
                )
            else:
                Z_better = 0

            if "S" in quantities:
                S_approx_log = calculate_log_S(U_approx, Z_better)
            else:
                S_approx_log = 0

            end = time.perf_counter()
            time_variables = end - start
            times_variables[i][j] = time_variables

            start = time.perf_counter()
            if "Z" in quantities:
                results["Z"][f"l0 = {l0}"][T] = (
                    abs((Z - Z_better) / Z) * 100
                )

            if "S" in quantities:
                results["S"][f"l0 = {l0}"][T] = (
                    abs((S_exact - S_approx_log) / S_exact) * 100
                )
            if "U" in quantities:
                results["U"][f"l0 = {l0}"][T] = (
                    abs((U_exact - U_approx) / U_exact) * 100
                )
            if "marginals" in quantities:
                results["marginals"][f"l0 = {l0}"][T] = (
                    compute_difference_general(
                        marginals_approx, marginals_exact, percent=True
                    )
                    * 100
                )
            if printing:
                print("computing ", filename)
                print(f"T = {T}")
                print(f"l0 = {l0}")
                print("error log   : ", abs(S_approx_log - S_exact) * 100 / S_exact)
                print("better Z ", Z_better)
                print("exact  Z ", Z)
        avg_times = np.sum(times_variables, axis=0)

        avg_times = avg_times / avg_times[0]

        for l0 in l_0_values:
            times_precalc[l0] /= times_precalc[0]

        latex_times = times_graph(avg_times=avg_times)
        latex_times_precalc = times_graph_precalc(times_precalc)
        latex = results_to_latex(results, observables=quantities)
        if not only_runtime:
            with open(f".\output\{filename}.tex", "w") as f:
                f.write(latex)
                f.write(latex_times_precalc)
                f.write(latex_times)
        else:
            with open(f".\output\{filename}_runtime.tex", "w") as f:
                f.write(latex_times_precalc)
                f.write(latex_times)


def precompute_values_only(
    adj,
    filename,
    l_0_values=[0, 1, 2, 3, 4, 5, 6],
    T_values=[(x + 1) * 0.25 for x in range(16)],
    check_precomputed=True,
):
    cache = load_cache()

    for i, T in enumerate(T_values):
        b = 1 / T
        tn = build_tn_from_adj(adj, beta=b)
        key = make_key(adj, T)
        if key in cache and check_precomputed:
            print("Loading exact values from cache...")
            Z, S_exact, U_exact, marginals_exact = cache[key]
            print("partition function", Z)
        else:
            print("calculating exact values")
            Z = tn.contract()
            print("partition function", Z)
            print("calculating S_exact")
            S_exact = calculate_S_exact(tn)
            print("exact S computed")
            U_exact = compute_energy_exact_new(tn, Z)
            # print(f"exact U calculated {U_exact}")
            # U_exact = compute_energy_exact_general(tn, Z)
            print(f"exact U calculated {U_exact}")
            marginals_exact = compute_marginals_exact_general(tn, Z)
            print("computed marginals")
            cache[key] = (Z, S_exact, U_exact, marginals_exact)
            save_cache(cache)
            print("saved cache")
