from itertools import chain
import math

import numpy as np
import quimb.tensor as qtn
import quimb as qu
from multiprocessing import Pool, Process, cpu_count

from collections import Counter

from math import floor
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from compute_values import (
    compute_energy,
    compute_energy_exact,
    compute_marginals,
    compute_marginals_exact,
    factors,
    tensors_that_share_edge,
    tensors_that_share_neighbourhood,
)



def compute_difference(size, approx, exact, percent=False):
    avg_distance = 0
    for i in range(size):
        for j in range(size):
            if percent:
                difference_tensor = (approx[(i, j)] - exact[(i, j)]) / exact[(i, j)]
            else:
                difference_tensor = approx[(i, j)] - exact[(i, j)]

            local_distance = abs(np.sum(abs(difference_tensor.data)))

            avg_distance += local_distance
    avg_distance /= size * size
    return avg_distance


def compute_difference_tvd(size, approx, exact):
    avg_distance = 0
    for i in range(size):
        for j in range(size):
            tvd = 0.5 * np.sum(np.abs(approx[(i, j)].data - exact[(i, j)].data))

            avg_distance += tvd
    avg_distance /= size * size
    return avg_distance


def compute_difference_tvd_only_middle(size, approx, exact, blockSize):
    avg_distance = 0
    offset = (blockSize - 1) / 2
    for i in range(floor(size / blockSize)):
        for j in range(floor(size / blockSize)):
            point = ((i * blockSize + offset), (j * blockSize + offset))
            # print("Point", point)
            tvd = 0.5 * np.sum(np.abs(approx[point].data - exact[point].data))

            avg_distance += tvd
    avg_distance /= floor(size / blockSize) * floor(size / blockSize)
    return avg_distance


def printResults(size, avg_diff_tvd, avg_diff_tvd_middle, algoName, iter):
    space = ""
    if iter < 10:
        space = " "
    print(
        f"{algoName} Size: {size} - avg diff TVD after {iter}{space} steps: {abs(avg_diff_tvd):.32f} "
    )
    space = ""
    if iter < 10:
        space = " "
    print(
        f"{algoName} Size: {size} - avg diff MID after {iter}{space} steps: {abs(avg_diff_tvd_middle):.32f} "
    )
    if iter < 10:
        space = " "
    print(
        f"{algoName} ------- All / MID = {abs(avg_diff_tvd / avg_diff_tvd_middle):.32f}"
    )
    print("---")


def add_result_dicts_SPECIFIC(res1, res2):
    """Add the distances of two result dicts."""
    out = {}
    for method, data in res1.items():
        out[method] = {"tvd all": []}
        res2_data = res2.get(method, {"tvd all": []})

        # Build a lookup for (input_size, iterations) -> distance in res2
        lookup = {
            (d["input_size"], d["iterations"]): d["distance"]
            for d in res2_data["tvd all"]
        }

        for entry in data["tvd all"]:
            key = (entry["input_size"], entry["iterations"])
            dist2 = lookup.get(key, 0.0)  # default 0 if missing
            out[method]["tvd all"].append(
                {
                    "input_size": entry["input_size"],
                    "iterations": entry["iterations"],
                    "distance": entry["distance"] + dist2,
                }
            )
    return out


def test_add_result_dicts():
    res1 = {
        "BlockBP size3": {
            "tvd all": [
                {"input_size": 15, "iterations": 2, "distance": np.float64(0.1)},
                {"input_size": 15, "iterations": 3, "distance": np.float64(0.2)},
            ]
        }
    }

    res2 = {
        "BlockBP size3": {
            "tvd all": [
                {"input_size": 15, "iterations": 2, "distance": np.float64(1.0)},
                {"input_size": 15, "iterations": 3, "distance": np.float64(2.0)},
            ]
        }
    }

    expected = {
        "BlockBP size3": {
            "tvd all": [
                {"input_size": 15, "iterations": 2, "distance": np.float64(1.1)},
                {"input_size": 15, "iterations": 3, "distance": np.float64(2.2)},
            ]
        }
    }

    result = add_result_dicts_SPECIFIC(res1, res2)
    assert np.allclose(
        result["BlockBP size3"]["tvd all"][0]["distance"],
        expected["BlockBP size3"]["tvd all"][0]["distance"],
    )
    assert np.allclose(
        result["BlockBP size3"]["tvd all"][1]["distance"],
        expected["BlockBP size3"]["tvd all"][1]["distance"],
    )
    result = add_result_dicts_SPECIFIC(res1, {})
    assert np.allclose(
        result["BlockBP size3"]["tvd all"][0]["distance"],
        res1["BlockBP size3"]["tvd all"][0]["distance"],
    )
    assert np.allclose(
        result["BlockBP size3"]["tvd all"][1]["distance"],
        res1["BlockBP size3"]["tvd all"][1]["distance"],
    )
    print("✅ test passed")


# Run test
def divideResultstestSpecific(old, divider):
    for algo in old.keys():
        for i in range(len(old[algo]["tvd all"])):
            old[algo]["tvd all"][i]["distance"] = (
                old[algo]["tvd all"][i]["distance"] / divider
            )
    return old



def create_latex_graph(
    results, output_filename="comparison_plot.pgf", x_axis="iterations"
):
    """
    Creates a LaTeX PGF plot comparing average distances to exact marginals.

    Parameters:
        results (dict): Measurement outcomes.
        output_filename (str): Output filename (.pgf) to save the plot.
        x_axis (str): 'input_size' or 'iterations'
    """
    assert x_axis in ("input_size", "iterations"), (
        "x_axis must be 'input_size' or 'iterations'"
    )

    # Use PGF backend for LaTeX
    mpl.use("pgf")
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    plt.figure(figsize=(6, 4))

    markers = ["o", "s", "D", "^", "v", "x"]
    linestyles = ["-", "--"]
    color_map = plt.cm.get_cmap("tab10")

    algorithms = list(results.keys())
    test_cases = list(next(iter(results.values())).keys())

    for i, alg in enumerate(algorithms):
        for j, test_case in enumerate(test_cases):
            entries = results[alg][test_case]
            entries_sorted = sorted(entries, key=lambda d: d[x_axis])
            x_vals = [d[x_axis] for d in entries_sorted]
            y_vals = [d["distance"] for d in entries_sorted]
            label = f"{alg} ({test_case})"
            plt.plot(
                x_vals,
                y_vals,
                label=label,
                marker=markers[j % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                color=color_map(i),
            )

    xlabel = "Iterations" if x_axis == "iterations" else "Input Size"
    plt.xlabel(xlabel)
    plt.ylabel("Avg. Distance to Exact Marginals")
    plt.title("Comparison of Marginal Estimation Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()


def export_to_pgfplots_table(results, filename, x_axis="iterations"):
    """
    Exports results as pgfplots-compatible LaTeX code.

    """

    print("results plot", results)

    algorithms = results.keys()
    test_cases = next(iter(results.values())).keys()

    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[%",
        f"    xlabel={{\\texttt{{{x_axis}}}}},"
        "    ylabel={Avg. Distance to Exact Marginals},",
        "    legend pos=north east,",
        "    grid=major",
        "]",
    ]

    for alg in algorithms:
        for case in test_cases:
            entries = sorted(results[alg][case], key=lambda d: d[x_axis])
            coords = " ".join([f"({d[x_axis]},{d['distance']})" for d in entries])
            lines.append(f"\\addplot coordinates {{{coords}}};")
            lines.append(f"\\addlegendentry{{{alg} ({case})}}")

    lines.append("\\end{axis}")
    lines.append("\\end{tikzpicture}")

    latex_code = "\n".join(lines)
    with open(f"{filename}.tex", "w") as f:
        f.write(latex_code)

    print("Exported to pgfplots_output.tex")


def merge_add_dicts(d1, d2):
    result = {}
    keys = set(d1) | set(d2)

    for key in keys:
        v1 = d1.get(key)
        v2 = d2.get(key)

        if isinstance(v1, dict) and isinstance(v2, dict):
            result[key] = merge_add_dicts(v1, v2)
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            result[key] = v1 + v2
        else:
            result[key] = (
                v1 if v2 is None else v2 if v1 is None else v2
            )  # Prefer d2's value
    return result


def divide_numeric_values(d, divisor):
    if isinstance(d, dict):
        return {k: divide_numeric_values(v, divisor) for k, v in d.items()}
    elif isinstance(d, list):
        return [divide_numeric_values(item, divisor) for item in d]
    elif isinstance(d, (float)):
        return d / divisor
    else:
        return d


def generate_latex_plot_ising(data: dict) -> str:
    """
    Generates LaTeX code using PGFPlots to visualize the algorithm comparison.

    Args:
        data (dict): Nested dictionary of the form:
                     {'algorithm': {'T': value, ...}, ...}

    Returns:
        str: LaTeX code string with PGFPlots diagram.
    """
    latex_code = [
        r"\begin{tikzpicture}",
        r"  \begin{axis}[",
        r"    xlabel={$T$},",
        r"    ylabel={Result Value},",
        r"    legend pos=north east,",
        r"    ymode=log,",
        r"    grid=major,",
        r"    width=12cm,",
        r"    height=8cm",
        r"  ]",
    ]

    colors = ["blue", "red", "green", "orange", "purple", "cyan"]
    for i, (algorithm, results) in enumerate(data.items()):
        sorted_items = sorted((float(k), float(v)) for k, v in results.items())
        coordinates = " ".join([f"({t}, {v:.2e})" for t, v in sorted_items])
        latex_code.extend(
            [
                rf"    \addplot+[mark=*, color={colors[i % len(colors)]}] coordinates {{ {coordinates} }};",
                rf"    \addlegendentry{{{algorithm}}}",
            ]
        )

    latex_code.extend([r"  \end{axis}", r"\end{tikzpicture}"])

    return "\n".join(latex_code)

