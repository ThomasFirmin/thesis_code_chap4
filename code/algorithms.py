from zellij.core import (
    IThreshold,
    BooleanStop,
)
from zellij.strategies.fractals import (
    PHS,
    ILS,
    DirectSampling,
    DBA,
    DBADirect,
    CenterSOO,
    Center,
)
from zellij.strategies.fractals.ils import RandomILSLHS

from zellij.strategies.tools import (
    Hypersphere,
    Direct,
    Section,
    DistanceToTheBest,
    DistanceToTheBestCentered,
    MoveUp,
    PotentiallyOptimalRectangle,
    LocallyBiasedPOR,
    AdaptivePOR,
    SooTreeSearch,
    BestFirstSearch,
    Min,
    Nothing,
    SigmaInf,
    Sigma2,
    NMSOSection,
    NMSOTreeSearch,
)

from zellij.strategies.fractals.dba import DBALHSUCBNew
from zellij.strategies.tools.geometry import LatinHypercubeUCB
from zellij.strategies.tools.scoring import UCB
from zellij.strategies.tools.tree_search import BaMSOOUCB


import numpy as np


def FDA(values, lf):
    sp = Hypersphere(values)

    explor = PHS(sp, inflation=1.75)
    exploi = ILS(sp, inflation=1.75)
    stop1 = BooleanStop(explor, "computed")
    stop2 = IThreshold(exploi, "step", 1e-16)
    dba = DBA(
        sp,
        MoveUp(sp, 5),
        (explor, stop1),
        (exploi, stop2),
        scoring=DistanceToTheBest(lf),
    )

    return dba


def FDA_D(values, lf):
    sp = Hypersphere(values)

    explor = PHS(sp, inflation=1.75)
    exploi = ILS(sp, inflation=1.75)
    stop1 = BooleanStop(explor, "computed")
    stop2 = IThreshold(exploi, "step", 1e-16)
    dba = DBA(
        sp,
        MoveUp(sp, 10),
        (explor, stop1),
        (exploi, stop2),
        scoring=DistanceToTheBest(lf),
    )

    return dba


def FDA_C(values, lf):
    sp = Hypersphere(values)

    explor = PHS(sp, inflation=1.75)
    exploi = ILS(sp, inflation=1.75)
    stop1 = BooleanStop(explor, "computed")
    stop2 = IThreshold(exploi, "step", 1e-8)
    dba = DBA(
        sp,
        MoveUp(sp, 5),
        exploration=(explor, stop1),
        exploitation=(exploi, stop2),
        scoring=DistanceToTheBestCentered(lf),
    )

    return dba


def DIRECT(values, lf):
    sp = Direct(values, measurement=SigmaInf())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBADirect(
        sp,
        PotentiallyOptimalRectangle(sp, int(68 * sp.size), maxopen=10000, error=1e-8),
        exploration=(explor, stop1),
        exploitation=(None, None),
        scoring=Nothing(),
    )

    return dba


def DIRECTL(values, lf):
    sp = Direct(values, measurement=SigmaInf())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBADirect(
        sp,
        LocallyBiasedPOR(sp, int(68 * sp.size), maxopen=10000, error=1e-16),
        exploration=(explor, stop1),
        exploitation=(None, None),
        scoring=Nothing(),
    )

    return dba


def DIRECTR(values, lf):
    sp = Direct(values, measurement=Sigma2())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBADirect(
        sp,
        AdaptivePOR(sp, int(68 * sp.size), maxopen=10000, error=1e-16),
        exploration=(explor, stop1),
        exploitation=(None, None),
        scoring=Nothing(),
    )

    return dba


def SOO(values, lf):
    sp = Section(values, section=3)

    explor = CenterSOO(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBA(
        sp,
        SooTreeSearch(sp, int(10 * np.sqrt((np.log(10**4 * sp.size)) ** 3))),
        (explor, stop1),
        (None, None),
        scoring=Min(),
    )

    return dba


def FDABfs(values, lf):
    sp = Hypersphere(values)

    explor = PHS(sp, inflation=1.75)
    exploi = ILS(sp, inflation=1.75)
    stop1 = BooleanStop(explor, "computed")
    stop2 = IThreshold(exploi, "step", 1e-16)
    dba = DBA(
        sp,
        BestFirstSearch(sp, 5, Q=sp.size),
        (explor, stop1),
        (exploi, stop2),
        scoring=DistanceToTheBest(lf),
    )

    return dba


def FDADBfs(values, lf):
    sp = Hypersphere(values)

    explor = PHS(sp, inflation=1.75)
    exploi = ILS(sp, inflation=1.75)
    stop1 = BooleanStop(explor, "computed")
    stop2 = IThreshold(exploi, "step", 1e-16)
    dba = DBA(
        sp,
        BestFirstSearch(sp, 10, Q=sp.size),
        (explor, stop1),
        (exploi, stop2),
        scoring=DistanceToTheBest(lf),
    )

    return dba


def DIRECTBfs(values, lf):
    sp = Direct(values, measurement=SigmaInf())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBADirect(
        sp,
        BestFirstSearch(sp, int(34 * sp.size)),
        exploration=(explor, stop1),
        exploitation=(None, None),
        scoring=Nothing(),
    )

    return dba


def SOOBfs(values, lf):
    sp = Section(values, section=3)

    explor = CenterSOO(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBA(
        sp,
        BestFirstSearch(
            sp, int(10 * np.sqrt((np.log(10**4 * sp.size)) ** 3)), Q=sp.size
        ),
        (explor, stop1),
        (None, None),
        scoring=Min(),
    )

    return dba


def NMSO(values, lf):
    sp = NMSOSection(values, section=3)

    explor = CenterSOO(sp)
    stop1 = BooleanStop(explor, "computed")

    basket = 25 * sp.size
    alpha = 1e-8 * sp.size
    beta = 1e-8 * sp.size

    dba = DBA(
        sp,
        NMSOTreeSearch(
            sp, min(int(34 * sp.size), 600), V=basket, alpha=alpha, beta=beta
        ),
        (explor, stop1),
        (None, None),
        scoring=Min(),
    )

    return dba


###################################


def UCBLHSVLowMI(values, lf):

    eps = 1e-4
    gsize = 2
    nu = 0.5

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSLowMI(values, lf):

    eps = 1e-4
    gsize = 6
    nu = 0.5

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSHighMI(values, lf):
    eps = 1e-4
    gsize = 10
    nu = 0.5

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSVHighMI(values, lf):
    eps = 1e-4
    gsize = 20
    nu = 0.5

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


###################################


def UCBLHSVLowLI(values, lf):

    eps = 1e-4
    gsize = 2
    nu = 0.05

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSLowLI(values, lf):

    eps = 1e-4
    gsize = 6
    nu = 0.05

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSHighLI(values, lf):
    eps = 1e-4
    gsize = 10
    nu = 0.05

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSVHighLI(values, lf):
    eps = 1e-4
    gsize = 20
    nu = 0.05

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


###################################


def UCBLHSVLowHI(values, lf):

    eps = 1e-4
    gsize = 2
    nu = 0.99

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSLowHI(values, lf):

    eps = 1e-4
    gsize = 6
    nu = 0.99

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSHighHI(values, lf):
    eps = 1e-4
    gsize = 10
    nu = 0.99

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


def UCBLHSVHighHI(values, lf):
    eps = 1e-4
    gsize = 20
    nu = 0.99

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    sample = Center(sp)

    dba = DBALHSUCBNew(sp, BaMSOOUCB(sp, levels, nu=nu), sample, scoring=UCB())
    return dba


###################################


def RandomILS(values, lf):
    eps = 1e-4
    gsize = 20
    nu = 0.99

    eps = 1 / eps
    levels = int(np.ceil(np.log(eps) / np.log(gsize)))
    if 1 / gsize**levels < eps:
        levels -= 1

    sp = LatinHypercubeUCB(values, levels, grid_size=gsize, strength=1)

    dba = RandomILSLHS(sp, int(10**4 * sp.size * 0.70))
    dba.radius = 0.125
    dba.step = 0.125
    return dba
