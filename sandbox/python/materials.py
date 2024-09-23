import os


import ikarus as iks
from ikarus import materials, utils

import numpy as np
import pandas as pd
import scipy as sp


from helpers import prettyprint


# mat = iks.materials.NeoHooke(E=1000, nu=0.3)
# matPS = mat.asPlaneStrain()

mat = iks.materials.StVenantKirchhoff(E=1000, nu=0.3)
matPS = mat.asPlaneStrain()

C = [0.600872, 0.859121, -0.358166]
C3d = [0.600872, 0.859121, 1, 0, 0, -0.358166]


e = matPS.storedEnergy(materials.StrainTags.rightCauchyGreenTensor, C)
S = matPS.stresses(materials.StrainTags.rightCauchyGreenTensor, C)
CT = matPS.tangentModuli(materials.StrainTags.rightCauchyGreenTensor, C)

# S3d = mat.stresses(materials.StrainTags.rightCauchyGreenTensor, C3d)

print(e)
prettyprint(S)
# prettyprint(S3d)

e = mat.storedEnergy(materials.StrainTags.rightCauchyGreenTensor, C3d)
print(e)