import os


import ikarus as iks
import ikarus.assembler
import ikarus.dirichlet_values
import ikarus.finite_elements
import ikarus.utils
import numpy as np
import pandas as pd
import scipy as sp

import dune.grid
import dune.functions
from dune.common import FieldVector

from helpers import prettyprint


def main():
    reader = (
        dune.grid.reader.gmsh,
        os.path.join(os.path.dirname(__file__), "../geometries/2d.msh"),
    )
    grid = dune.grid.ugGrid(reader, dimgrid=2)

    basisLagrange1 = iks.basis(
        grid,
        dune.functions.Power(dune.functions.Lagrange(order=1), 2, layout="interleaved"),
    )

    flatBasis = basisLagrange1.flat()

    lambdaLoad = iks.Scalar(10.0)

    mat = iks.materials.StVenantKirchhoff(E=1000, nu=0.499)
    matPS = mat.asPlaneStrain()

    # mat = iks.materials.NeoHooke(E=1000, nu=0.0)
    # matPS = mat.asPlaneStrain()

    def neumannLoad(x, lambdaVal):
        return np.array([0, -lambdaVal])

    neumannVertices = np.zeros(grid.size(2), dtype=bool)
    def loadTopEdgePredicate(x):   
        return True if x[1] == 1.0 else False

    indexSet = grid.indexSet
    for v in grid.vertices:
        neumannVertices[indexSet.index(v)] = loadTopEdgePredicate(v.geometry.center)

    boundaryPatch = iks.utils.boundaryPatch(grid, neumannVertices)
    nBLoad = iks.finite_elements.neumannBoundaryLoad(boundaryPatch, neumannLoad)


    nonLinEalstic = iks.finite_elements.nonLinearElastic(matPS)
    fes = []

    for e in grid.elements:
        fes.append(iks.finite_elements.makeFE(basisLagrange1, nonLinEalstic, nBLoad))
        fes[-1].bind(e)

    dirichletValues = iks.dirichletValues(flatBasis)

    def fixBottom(vec, localIndex, localView, intersection):
        if intersection.geometry.center[1] == -1.0:
            vec[localView.index(localIndex)] = True

    dirichletValues.fixBoundaryDOFs(fixBottom)

    assembler = iks.assembler.sparseFlatAssembler(fes, dirichletValues)
    dRed = np.zeros(assembler.reducedSize())

    feReq = fes[0].createRequirement()

    def gradient(dRedInput):
        feReq = fes[0].createRequirement()
        feReq.insertParameter(lambdaLoad)
        dBig = assembler.createFullVector(dRedInput)
        feReq.insertGlobalSolution(dBig)
        return assembler.vector(
            feReq, iks.VectorAffordance.forces, iks.DBCOption.Reduced
        )
    def hess(dRedInput):
        feReq = fes[0].createRequirement()
        feReq.insertParameter(lambdaLoad)
        dBig = assembler.createFullVector(dRedInput)
        feReq.insertGlobalSolution(dBig)
        return assembler.matrix(
            feReq, iks.MatrixAffordance.stiffness, iks.DBCOption.Reduced
        ).todense()  # this is slow, but for this test we don't care
    
    d = sp.optimize.root(gradient, jac=hess, x0=dRed, tol=1e-10)

    #print(d)
    assert d.success
    prettyprint(d.x, 8)
    print("")

    d = assembler.createFullVector(d.x)
    feReq.insertGlobalSolution(d)
    res1 = fes[0].calculateAt(feReq, np.array([0.5, 0.5]), "PK2Stress")

    prettyprint(iks.utils.fromVoigt(res1))

if __name__ == "__main__":
    main()

