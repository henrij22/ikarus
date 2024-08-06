# SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later


# import debug_info

# debug_info.setDebugFlags()

import ikarus as iks
from ikarus import finite_elements, utils, assembler
from dune.iga.basis import defaultGlobalBasis, Power, Nurbs

import numpy as np
import pandas as pd

import dune.grid
# import dune.functions

#from dune.vtk import vtkWriter

import os
from dune.iga import ControlPoint, ControlPointNet, NurbsPatchData, IGAGrid

from dune.common.hashit import hashIt
from dune.iga.basis import preBasisTypeName
from dune.generator.generator import SimpleGenerator

pd.options.display.float_format = '{:.3f}'.format
def prittyprint(array):
    df = pd.DataFrame(array)
    print(df)


def globalBasis(gv, tree):
    generator = SimpleGenerator("BasisHandler", "Ikarus::Python")

    pbfName = preBasisTypeName(tree, gv.cppTypeName)
    element_type = f"Ikarus::BasisHandler<{pbfName}>"
    includes = []
    includes += list(gv.cppIncludes)
    includes += ["dune/iga/nurbsbasis.hh"]
    includes += ["ikarus/python/basis/basis.hh"]

    moduleName = "Basis_" + hashIt(element_type)
    module = generator.load(
        includes=includes, typeName=element_type, moduleName=moduleName
    )
    basis = defaultGlobalBasis(gv, tree)
    return module.BasisHandler(basis)


def lagrange():
    reader = (dune.grid.reader.gmsh, os.path.join(os.path.dirname(__file__), "2d.msh"))
    grid = dune.grid.ugGrid(reader, dimgrid=2)

    basisLagrange1 = iks.basis(
        grid,
        dune.functions.Power(dune.functions.Lagrange(order=2), 2, layout="interleaved"),
    )

    flatBasis = basisLagrange1.flat()
    d = np.zeros(len(flatBasis))

    lambdaLoad = iks.ValueWrapper(1.0)

    fes = []

    linElastic = iks.finite_elements.linearElastic(youngs_modulus=1000, nu=0.3)
    for e in grid.elements:
        fes.append(iks.finite_elements.makeFE(basisLagrange1, linElastic))
        fes[-1].bind(e)

    print(len(fes))
    req = fes[0].createRequirement()
    req.insertParameter(lambdaLoad)
    req.insertGlobalSolution(d)

    stiffness = np.zeros((18, 18))
    fes[0].calculateMatrix(req, iks.MatrixAffordance.stiffness, stiffness)

    print(stiffness)


def iga():
    cp = ControlPoint((0, 0), 1)
    cp2 = ControlPoint((1, 0), 1)
    cp3 = ControlPoint((0, 2), 1)
    cp4 = ControlPoint((1, 2), 1)

    netC = ((cp, cp2), (cp3, cp4))
    net = ControlPointNet(netC)

    nurbsPatchData = NurbsPatchData(((0, 0, 1, 1), (0, 0, 1, 1)), net, (1, 1))
    nurbsPatchData = nurbsPatchData.degreeElevate(0, 1)
    nurbsPatchData = nurbsPatchData.degreeElevate(1, 1)

    for i in range(0, len(nurbsPatchData.controlPoints)):
        cp = nurbsPatchData.controlPoints.directGet(i)
        print(cp.coords)


    grid = IGAGrid(nurbsPatchData)
    grid.hierarchicalGrid.globalRefine(1)


    basis = globalBasis(grid, Power(Nurbs(), 2, layout="interleaved"))

    flatBasis = basis.flat()
    d = np.zeros(len(flatBasis))

    lambdaLoad = iks.Scalar(1.0)

    def vL(x, lambdaVal):
        return np.array([10.0, 0])
    
    vLoad = iks.finite_elements.volumeLoad2D(vL)

    fes = []

    linElastic = iks.finite_elements.linearElastic(youngs_modulus=1000, nu=0.3)
    for e in grid.elements:
        fes.append(iks.finite_elements.makeFE(basis, linElastic, vLoad))
        fes[-1].bind(e)

    print(len(fes))
    req = fes[0].createRequirement()
    req.insertParameter(lambdaLoad)
    req.insertGlobalSolution(d)

    dirichletValues = iks.dirichletValues(flatBasis)

    def fixLeftHandEdge(vec, localIndex, localView, intersection):
        if intersection.geometry.center[0] == 0.0:
            vec[localView.index(localIndex)] = True

    dirichletValues.fixBoundaryDOFs(fixLeftHandEdge)
    assert dirichletValues.fixedDOFsize > 0

    assembler = iks.assembler.denseFlatAssembler(fes, dirichletValues)
    assembler.bind(req, iks.AffordanceCollection.elastoStatics, iks.DBCOption.Full)
    
    K = assembler.matrix()
    F = assembler.vector()

    prittyprint(K)
    print(np.linalg.norm(K))

    # So something
    D = np.linalg.solve(K, F)
    print(np.linalg.norm(D))
    fx = flatBasis.asFunction(D)

    # vtkWriter(grid, "test", pointData={("displacement", (0, 1)): fx})

    vtkWriter = grid.trimmedVtkWriter(8)
    vtkWriter.addPointData(fx, name="displacement")
    vtkWriter.write("test")

if __name__ == "__main__":
    iga()
