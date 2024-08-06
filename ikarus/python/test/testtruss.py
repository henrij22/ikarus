# SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later


# import debug_info

# debug_info.setDebugFlags()

import ikarus as iks
from ikarus import finite_elements, assembler
import numpy as np
import scipy as sp
import os
import pandas as pd

import dune.foamgrid
import dune.grid
import dune.functions

pd.options.display.float_format = '{:.3f}'.format
def prittyprint(array):
    df = pd.DataFrame(array)
    print(df)

def truss():

    E = 1
    A = 1

    gridDim = 1
    worldDim = 2

    filedir = os.path.dirname(__file__)
    filename = os.path.join(filedir, "truss.msh")
    reader = (dune.grid.reader.gmsh, filename)

    grid = dune.foamgrid.foamGrid(reader, gridDim, worldDim)
    trusses = finite_elements.truss(youngs_modulus=E, cross_section=A)

    basis = iks.basis(
        grid, dune.functions.Power(dune.functions.Lagrange(order=1), worldDim)
    )

    fes = []


    for e in grid.elements:
        fes.append(finite_elements.makeFE(basis, trusses))
        fes[-1].bind(e)


    lambdaLoad = iks.Scalar(0.0)
    flatBasis = basis.flat()
    d = np.zeros(len(flatBasis))

    req = fes[0].createRequirement()
    req.insertParameter(lambdaLoad)
    req.insertGlobalSolution(d)

    stiffness = np.zeros((worldDim * 2, worldDim * 2))
    fes[0].calculateMatrix(req, iks.MatrixAffordance.stiffness, stiffness)

    prittyprint(stiffness)


if __name__ == "__main__":
    truss()
