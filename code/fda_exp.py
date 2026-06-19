#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence produces only rudimentary progress messages and does not provide
batch distribution or timing prints, as `example_experiment2.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

>>> import cma  # doctest:+SKIP
>>> def fmin(fun, x0):
...     return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import cocoex

from zellij.core import (
    ArrayVar,
    FloatVar,
    Loss,
    Minimizer,
    Experiment,
    Calls,
    BooleanStop,
)

from zellij.utils.converters import FloatMinMax, ArrayDefaultC
from zellij.strategies.fractals import ILSLHS
from zellij.core import IThreshold

from algorithms import (
    FDA,
    FDA_D,
    FDA_C,
    DIRECT,
    DIRECTL,
    DIRECTR,
    SOO,
    NMSO,
    FDABfs,
    FDADBfs,
    DIRECTBfs,
    SOOBfs,
    UCBLHSVLowLI,
    UCBLHSVLowMI,
    UCBLHSVLowHI,
    UCBLHSLowLI,
    UCBLHSHighLI,
    UCBLHSVHighLI,
    UCBLHSLowMI,
    UCBLHSHighMI,
    UCBLHSVHighMI,
    UCBLHSLowHI,
    UCBLHSHighHI,
    UCBLHSVHighHI,
    RandomILS,
)

import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--fct", type=int, default=None)
parser.add_argument("--inst", type=int, default=None)
parser.add_argument("--dim", type=int, default=None)
# parser.add_argument("--deltaf", type=float, default=10e-4)
parser.add_argument("--alg", type=str, default="FDA")
parser.add_argument("--extraexp", type=int, default=False)

parser.set_defaults(gpu=True, record_time=True)

args = parser.parse_args()
fct = args.fct
inst = args.inst
dim = args.dim
# deltaf = args.deltaf
alg = args.alg
extraexp = args.extraexp


def objective(func):
    def loss(x):
        res = func(x)
        return {"obj": res}

    return loss


if alg == "FDA":
    dba = FDA
elif alg == "FDA-D":
    dba = FDA_D
elif alg == "FDA-C":
    dba = FDA_C
elif alg == "DIRECT":
    dba = DIRECT
elif alg == "DIRECT-L":
    dba = DIRECTL
elif alg == "DIRECT-R":
    dba = DIRECTR
elif alg == "SOO":
    dba = SOO
elif alg == "NMSO":
    dba = NMSO
elif alg == "FDA-BFS":
    dba = FDABfs
elif alg == "FDAD-BFS":
    dba = FDADBfs
elif alg == "DIRECT-BFS":
    dba = DIRECTBfs
elif alg == "SOO-BFS":
    dba = SOOBfs
#####################################
elif alg == "UCBLHSVLowLI":
    dba = UCBLHSVLowLI
elif alg == "UCBLHSLowLI":
    dba = UCBLHSLowLI
elif alg == "UCBLHSHighLI":
    dba = UCBLHSHighLI
elif alg == "UCBLHSVHighLI":
    dba = UCBLHSVHighLI
elif alg == "UCBLHSVLowMI":
    dba = UCBLHSVLowMI
elif alg == "UCBLHSLowMI":
    dba = UCBLHSLowMI
elif alg == "UCBLHSHighMI":
    dba = UCBLHSHighMI
elif alg == "UCBLHSVHighMI":
    dba = UCBLHSVHighMI
elif alg == "UCBLHSVLowHI":
    dba = UCBLHSVLowHI
elif alg == "UCBLHSLowHI":
    dba = UCBLHSLowHI
elif alg == "UCBLHSHighHI":
    dba = UCBLHSHighHI
elif alg == "UCBLHSVHighHI":
    dba = UCBLHSVHighHI
elif alg == "RandomILS":
    dba = RandomILS

# #####################################

### input
suite_name = "bbob"
output_folder = alg
budget_multiplier = 10**4  # increase to 10, 100, ...

### prepare
options = ""
if dim is not None:
    options += f"dimensions: {dim} "
if fct is not None:
    options += f"function_indices: {fct+1} "
if inst is not None:
    options += f"instance_indices: {inst+1} "

suite = cocoex.Suite(
    suite_name,
    "",
    options,
)

observer = cocoex.Observer(
    suite_name,
    f"result_folder: {output_folder}/{output_folder}_{dim}_{fct}_{inst}_ algorithm_name: {output_folder}",
)
minimal_print = cocoex.utilities.MiniPrint()

### go
for idx1, problem in enumerate(suite):
    problem.observe_with(observer)  # generates the data for cocopp post-processing

    # print(
    #     f"{alg} > Problem n° : {problem.index}/{len(suite)}\n{alg} > Problem dimension : {problem.dimension}"
    # )

    ###############
    # ZELLIJ PART #
    ###############

    lf = Loss(
        objective=[Minimizer("obj")],
    )(objective(problem))

    values = ArrayVar(converter=ArrayDefaultC())
    for idx, (l, u) in enumerate(zip(problem.lower_bounds, problem.upper_bounds)):
        values.append(FloatVar(f"float_{idx}", l, u, converter=FloatMinMax()))

    budget = int(problem.dimension * budget_multiplier)
    if extraexp:
        stop = Calls(lf, int(budget * 0.70)) | BooleanStop(problem, "final_target_hit")
    else:
        stop = Calls(lf, budget) | BooleanStop(problem, "final_target_hit")

    optimizer = dba(values=values, lf=lf)
    exp = Experiment(optimizer, lf, stop, verbose=False)
    exp.run()

    if extraexp:
        length = 0.125
        i = 0
        fts = optimizer.tree_search.levels_bf[-2:0:-1]
        yts = optimizer.tree_search.levels_bests[-2:0:-1]

        atleastonenotnone = False

        stop = False
        while not stop:
            f = fts[i]
            if f is not None:
                y = yts[i]
                x = (f.upper + f.lower) / 2
                exploi = ILSLHS(optimizer.search_space)
                stop1 = (
                    IThreshold(exploi, "step", 1e-16)
                    | Calls(lf, budget)
                    | BooleanStop(problem, "final_target_hit")
                )
                exploi.current_point = x
                exploi.current_score = y
                exploi.step = length
                exp = Experiment(exploi, lf, stop1, verbose=False)
                exp.run()
                stop = stop1()
                atleastonenotnone = True
            else:
                stop = False

            i = (i + 1) % len(fts)
            if i == 0:
                length /= 2
                if atleastonenotnone and not stop:
                    stop = False
                else:
                    stop = True

    minimal_print(problem)
    gc.collect()
    ###################
    # END ZELLIJ PART #
    ###################
