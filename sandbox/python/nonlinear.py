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

from helpers import prittyprint


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
    d = np.zeros(len(flatBasis))

    lambdaLoad = iks.Scalar(1.0)

    fes = []

    # Kurzer EAS Exkurs
    mat = iks.materials.LinearElasticity(E=1000, nu=0.0)
    matPS = mat.asPlaneStrain()

    linElastic = iks.finite_elements.linearElastic(matPS)
    easF = iks.finite_elements.eas(4)

    for e in grid.elements:
        fes.append(iks.finite_elements.makeFE(basisLagrange1, linElastic, easF))
        fes[-1].bind(e)

    # # mat = iks.materials.StVenantKirchhoff(E=1000, nu=0.0)
    # # matPS = mat.asPlaneStrain()

    # mat = iks.materials.NeoHooke(E=1000, nu=0.0)
    # matPS = mat.asPlaneStrain()


    # nonLinEalstic = iks.finite_elements.nonLinearElastic(matPS)
    # for e in grid.elements:
    #     fes.append(iks.finite_elements.makeFE(basisLagrange1, nonLinEalstic))
    #     fes[-1].bind(e)

    req = fes[0].createRequirement()
    req.insertParameter(lambdaLoad)
    req.insertGlobalSolution(d)

    stiffness = np.zeros((8, 8))
    fes[0].calculateMatrix(req, iks.MatrixAffordance.stiffness, stiffness)

    prittyprint(stiffness)

    # # IBB Ordering??
    # d = np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=float)
    # # d =np.zeros(8)
    # req.insertGlobalSolution(d)

    # fes[0].calculateMatrix(req, iks.MatrixAffordance.stiffness, stiffness)
    # prittyprint(stiffness)
    # print(np.linalg.norm(stiffness))

    # forces = np.zeros(8)
    # fes[0].calculateVector(req, iks.VectorAffordance.forces, forces)
    # prittyprint(forces)


    # strains = [-0.375, 0, 0]
    # C = matPS.tangentModuli("greenLagrangian", strains)

    # prittyprint(C)



    # f = flatBasis.asFunction(d)

    # stress = fes[0].calculateAt(req, np.array([0.25, 0.25]), "PK2Stress")
    # print(stress)

    # x = FieldVector([0, 0])
    # for e in grid.elements:
    #     print(f(e, FieldVector([0, 0])))
    #     print(f(e, FieldVector([1, 0])))
    #     print(f(e, FieldVector([1, 1])))
    #     print(f(e, FieldVector([0, 1])))
        # print(e.geometry.volume)
        # for i in range(0, 4):
        #     print(e.geometry.corner(i))


if __name__ == "__main__":
    main()

