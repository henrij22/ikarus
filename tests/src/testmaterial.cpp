// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include "testhelpers.hh"

#include <dune/common/test/testsuite.hh>

#include <Eigen/Eigenvalues>

#include <ikarus/finiteelements/mechanics/materials.hh>
#include <ikarus/finiteelements/physicshelper.hh>
#include <ikarus/utils/functionsanitychecks.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/nonlinearoperator.hh>

using namespace Ikarus;
using Dune::TestSuite;

template <StrainTags strainTag>
double transformStrainAccordingToStrain(auto& e) {
  double strainDerivativeFactor = 1;

  if (strainTag == StrainTags::greenLagrangian or strainTag == StrainTags::linear) {
    e = ((e.transpose() + e + 3 * Eigen::Matrix3d::Identity()) / 10).eval();
    e /= e.array().maxCoeff();
    auto C = (2 * e + Eigen::Matrix3d::Identity()).eval();
    Eigen::EigenSolver<Eigen::Matrix3d> esC(C);
    e                      = 0.5 * (C / esC.eigenvalues().real().maxCoeff() - Eigen::Matrix3d::Identity());
    strainDerivativeFactor = 1;
  } else if (strainTag == StrainTags::rightCauchyGreenTensor) {
    e = (e.transpose() + e).eval();
    Eigen::EigenSolver<Eigen::Matrix3d> esC(e);
    e += (-esC.eigenvalues().real().minCoeff() + 1) * Eigen::Matrix3d::Identity();
    esC.compute(e);
    e /= esC.eigenvalues().real().maxCoeff();

    assert(esC.eigenvalues().real().minCoeff() > 0 &&
           " The smallest eigenvalue is negative this is unsuitable for the tests");

    strainDerivativeFactor = 0.5;
  } else if (strainTag == StrainTags::deformationGradient) {
    e = (e + 3 * Eigen::Matrix3d::Identity()).eval(); // create positive definite matrix
    e = e.sqrt();
  }
  return strainDerivativeFactor;
}

template <StrainTags strainTag, typename MaterialImpl>
auto testMaterialWithStrain(const MaterialImpl& mat, const double tol = 1e-13) {
  TestSuite t(mat.name() + " InputStrainMeasure: " + toString(strainTag));
  std::cout << "Test: " << t.name() << " started\n";

  Eigen::Matrix3d e;
  e.setRandom();

  double strainDerivativeFactor = transformStrainAccordingToStrain<strainTag>(e);

  auto ev = toVoigtAndMaybeReduce(e, mat, true);
  static_assert(MaterialImpl::isReduced or
                (decltype(ev)::RowsAtCompileTime == 6 and decltype(ev)::ColsAtCompileTime == 1));

  auto energy    = mat.template storedEnergy<strainTag>(e);
  auto energyV   = mat.template storedEnergy<strainTag>(ev);
  auto stressesV = mat.template stresses<strainTag>(e);

  auto stressesVV = mat.template stresses<strainTag>(ev);

  auto moduliV  = mat.template tangentModuli<strainTag>(e);
  auto moduliVV = mat.template tangentModuli<strainTag>(ev);

  t.check(Dune::FloatCmp::le(std::abs(energyV - energy), tol))
      << std::string("Energy obtained from matrix and from Voigt representation do not coincide \n") << energy
      << "\n and \n"
      << energyV << "\n Diff: " << energy - energyV << " with tol: " << tol;
  if constexpr (requires { mat.impl().template stressesImpl<false>(e); }) {
    auto stresses   = mat.template stresses<strainTag, false>(e);
    auto stressesVM = mat.template stresses<strainTag, false>(ev);
    t.check(isApproxSame(toVoigt(stresses, false), enlargeIfReduced<MaterialImpl>(stressesV), tol))
        << std::string("Voigt representation of stresses does not coincide with matrix representation \n")
        << toVoigt(stresses, false) << "\n and \n"
        << enlargeIfReduced<MaterialImpl>(stressesV) << "\n Diff: \n"
        << toVoigt(stresses, false) - enlargeIfReduced<MaterialImpl>(stressesV);
    t.check(isApproxSame(toVoigt(stressesVM, false), enlargeIfReduced<MaterialImpl>(stressesV), tol))
        << std::string(" stresses obtained with strains from voigt, does not coincide with matrix representation \n")
        << stressesVM << "\n and \n"
        << stressesV;
  }
  t.check(isApproxSame(stressesVV, stressesV, tol))
      << std::string(
             "Voigt representation of stresses obtained with strains from "
             "voigt, does not coincide with matrix representation \n")
      << stressesVV << "\n and \n"
      << stressesV << "\n Diff: \n"
      << stressesVV - stressesV;

  if constexpr (requires { mat.impl().template tangentModuliImpl<false>(e); }) {
    auto moduli = mat.template tangentModuli<strainTag, false>(e);
    if constexpr (!MaterialImpl::isReduced) {
      t.check(isApproxSame(toVoigt(moduli), moduliV, tol))
          << std::string("Voigt representation of tangent moduli does not coincide with Tensor<4> representation \n")
          << toVoigt(moduli) << "\n and \n"
          << moduliV;
    }

    t.check(isApproxSame(moduliVV, moduliV, tol)) << std::string(
                                                         "Voigt representation of tangent moduli obtained with Voigt "
                                                         "object does not coincide with tangent moduli "
                                                         "obtained with matrix object  \n")
                                                  << moduliVV << "\n and \n"
                                                  << moduliV;
  }

  // Vanishing strain implementation is not tested against checkGradient (off by 1.0)
  if constexpr (traits::isSpecializationNonTypeAndTypes<VanishingStrain, MaterialImpl>::value) {
    return t;
  }

  auto f  = [&](auto& xv) { return mat.template storedEnergy<strainTag>(xv); };
  auto df = [&](auto& xv) { return (mat.template stresses<strainTag>(xv) * strainDerivativeFactor).eval(); };

  auto ddf = [&](auto& xv) {
    return (mat.template tangentModuli<strainTag>(xv) * strainDerivativeFactor * strainDerivativeFactor).eval();
  };

  auto nonLinOp    = Ikarus::NonLinearOperator(functions(f, df, ddf), parameter(ev));
  auto subNonLinOp = nonLinOp.template subOperator<1, 2>();

  t.check(utils::checkGradient(nonLinOp, {.draw = false, .writeSlopeStatementIfFailed = true}))
      << std::string("checkGradient Failed");
  t.check(utils::checkHessian(nonLinOp, {.draw = false, .writeSlopeStatementIfFailed = true}))
      << std::string("checkHessian Failed");
  t.check(utils::checkJacobian(subNonLinOp, {.draw = false, .writeSlopeStatementIfFailed = true}))
      << std::string("checkJacobian Failed");

  return t;
}

template <typename Material>
auto testMaterial(Material mat) {
  TestSuite t("testMaterial");
  if constexpr (std::is_same_v<Material, LinearElasticity> or
                std::is_same_v<Material,
                               Ikarus::VanishingStress<std::array<Ikarus::Impl::StressIndexPair, 1>({{{2, 2}}}),
                                                       Ikarus::LinearElasticity>>) {
    t.subTest(testMaterialWithStrain<StrainTags::linear>(mat));
  } else if constexpr (std::is_same_v<Material,
                                      Ikarus::VanishingStress<std::array<Ikarus::Impl::StressIndexPair, 1>({{{2, 2}}}),
                                                              Ikarus::StVenantKirchhoff>>) {
    t.subTest(testMaterialWithStrain<StrainTags::greenLagrangian>(mat));
  } else {
    if constexpr (Material::isReduced) {
      t.subTest(testMaterialWithStrain<StrainTags::greenLagrangian>(mat, 1e-12));
      t.subTest(testMaterialWithStrain<StrainTags::rightCauchyGreenTensor>(mat, 1e-12));
    } else {
      t.subTest(testMaterialWithStrain<StrainTags::greenLagrangian>(mat));
      t.subTest(testMaterialWithStrain<StrainTags::rightCauchyGreenTensor>(mat));
    }
  }
  return t;
}

template <StrainTags strainTag, typename MaterialImpl>
auto testPlaneStrainAgainstPlaneStress(const double tol = 1e-10) {
  TestSuite t(MaterialImpl::name() + " InputStrainMeasure: " + toString(strainTag));
  std::cout << "TestPlaneStrinAgainstPlaneStress: " << t.name() << " started\n";

  Eigen::Matrix3d e;
  e.setRandom();

  transformStrainAccordingToStrain<strainTag>(e);

  // instantiate material models
  LamesFirstParameterAndShearModulus matPar{.lambda = 0, .mu = 1000}; // \nu = 0

  auto mat            = MaterialImpl{matPar};
  auto planeStrainMat = planeStrain(mat);
  auto planeStressMat = planeStress(mat);

  // energy should be the same for plane stress and plane strain
  auto energies = std::array<double, 2>{planeStrainMat.template storedEnergy<strainTag>(e),
                                        planeStressMat.template storedEnergy<strainTag>(e)};

  t.check(Dune::FloatCmp::le(std::abs(energies[0] - energies[1]), tol))
      << "Energies for plane strain and plane stress should be the same but are"
      << "\n"
      << energies[0] << " and " << energies[1] << "\n Diff: " << energies[0] - energies[1] << " with tol: " << tol;

  // Stresses should be the same
  auto stressPlaneStrain = planeStrainMat.template stresses<strainTag>(e);
  auto stressPlaneStress = planeStressMat.template stresses<strainTag>(e);

  t.check(isApproxSame(stressPlaneStrain, stressPlaneStress, tol))
      << "Stresses for plane strain and plane stress should be the same but are"
      << "\n"
      << stressPlaneStrain << "\nand\n " << stressPlaneStress << "\n Diff: " << stressPlaneStrain - stressPlaneStress
      << " with tol: " << tol;

  // If we compare the plain stress material tensor with plain strain material tensor it should be the same for nu = 0

  auto matTangent            = toVoigt(mat.template tangentModuliImpl<false>(e));
  auto matTangentPlaneStrain = toVoigt(planeStrainMat.template tangentModuliImpl<false>(e));
  auto matTangentPlaneStress = toVoigt(planeStressMat.template tangentModuliImpl<false>(e));

  t.check(isApproxSame(matTangentPlaneStrain, matTangentPlaneStress, tol))
      << "Material Tangent for plane strain and plane stress should be the same but are"
      << "\n"
      << matTangentPlaneStrain << "\nand\n " << matTangentPlaneStress
      << "\n Diff: " << matTangentPlaneStrain - matTangentPlaneStress << " with tol: " << tol;

  // Test upper block
  auto matTagentUpper            = matTangent.template block<2, 2>(0, 0);
  auto matTangentPlanStrainUpper = matTangentPlaneStrain.template block<2, 2>(0, 0);

  t.check(isApproxSame(matTagentUpper, matTangentPlanStrainUpper, tol))
      << "Uppper part of material tangent for 3d model and plane strain  should be the same but are"
      << "\n"
      << matTagentUpper << "\nand\n " << matTangentPlanStrainUpper
      << "\n Diff: " << matTagentUpper - matTangentPlanStrainUpper << " with tol: " << tol;

  return t;
}

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);
  TestSuite t;

  LamesFirstParameterAndShearModulus matPar{.lambda = 1000, .mu = 500};

  auto svk = StVenantKirchhoff(matPar);
  t.subTest(testMaterial(svk));

  auto nh = NeoHooke(matPar);
  t.subTest(testMaterial(nh));

  auto le = LinearElasticity(matPar);
  t.subTest(testMaterial(le));

  auto leRed = makeVanishingStress<Impl::StressIndexPair{2, 2}>(le, 1e-12);
  t.subTest(testMaterial(leRed));

  auto svkRed = makeVanishingStress<Impl::StressIndexPair{2, 2}>(svk, 1e-12);
  t.subTest(testMaterial(svkRed));

  auto nhRed = makeVanishingStress<Impl::StressIndexPair{2, 2}>(nh, 1e-12);
  t.subTest(testMaterial(nhRed));

  auto nhRed2 = makeVanishingStress<Impl::StressIndexPair{1, 1}, Impl::StressIndexPair{2, 2}>(nh, 1e-12);
  t.subTest(testMaterial(nhRed2));

  auto nhRed3 =
      makeVanishingStress<Impl::StressIndexPair{2, 1}, Impl::StressIndexPair{2, 0}, Impl::StressIndexPair{2, 2}>(nh,
                                                                                                                 1e-12);
  t.subTest(testMaterial(nhRed3));

  auto nhRed4 = shellMaterial(nh, 1e-12);
  t.subTest(testMaterial(nhRed4));

  auto nhRed5 = beamMaterial(nh, 1e-12);
  t.subTest(testMaterial(nhRed5));

  auto nhRed6 = planeStress(nh, 1e-12);
  t.subTest(testMaterial(nhRed6));

  auto svkPlaneStrain = planeStrain(svk);
  t.subTest(testMaterial(svkPlaneStrain));

  auto nhPlaneStrain = planeStrain(nh);
  t.subTest(testMaterial(nhPlaneStrain));

  auto linPlaneStrain = planeStrain(le);
  t.subTest(testMaterialWithStrain<StrainTags::linear>(linPlaneStrain));

  t.subTest(testPlaneStrainAgainstPlaneStress<StrainTags::greenLagrangian, StVenantKirchhoff>());
  // t.subTest(testPlaneStrinAgainstPlaneStress<StrainTags::rightCauchyGreenTensor, NeoHooke>());

  return t.exit();
}
