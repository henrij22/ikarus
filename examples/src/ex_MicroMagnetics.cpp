//
// Created by Alex on 21.07.2021.
//

#include "../../config.h"

#include <dune/alugrid/grid.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/iga/nurbsgrid.hh>

#include "spdlog/spdlog.h"

#include <Eigen/Core>

#include "ikarus/Controlroutines/LoadControl.h"
#include "ikarus/FiniteElements/Micromagnetics/MicroMangeticsWithVectorPotential.h"
#include "ikarus/Solver/NonLinearSolver/TrustRegion.hpp"
#include "ikarus/utils/Observer/controlVTKWriter.h"
#include "ikarus/utils/Observer/genericControlObserver.h"
#include "ikarus/utils/drawing/griddrawer.h"
#include <ikarus/Assembler/SimpleAssemblers.h>
#include <ikarus/LinearAlgebra/NonLinearOperator.h>
#include <ikarus/utils/functionSanityChecks.h>
#include <ikarus/utils/utils/algorithms.h>

constexpr int magnetizationOrder    = 1;
constexpr int vectorPotOrder        = 1;
constexpr int gridDim               = 2;
constexpr int directorDim           = 3;
constexpr int vectorPotDim          = gridDim == 2 ? 1 : 3;
constexpr int directorCorrectionDim = directorDim - 1;

using DirectorVector  = Dune::BlockVector<Ikarus::UnitVector<double, directorDim>>;
using VectorPotVector = Dune::BlockVector<Ikarus::RealTuple<double, vectorPotDim>>;
using MultiTypeVector = Dune::MultiTypeBlockVector<DirectorVector, VectorPotVector>;

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  using namespace Ikarus;
  Ikarus::FiniteElements::MagneticMaterial mat({.A = 1.0e-11, .K = 2e4, .ms = 1.432e6});
  //  Ikarus::FiniteElements::MagneticMaterial mat({.A = 2.0e-11, .K = 1e-3, .ms = 8e2});
  const double lx              = sqrt(2 * mat.A / (mat.mu0 * mat.ms * mat.ms));
  const double lengthUnit      = 1e-9;
  const double sizedom1InMeter = 30 * lengthUnit;
  const double sizedom1        = sizedom1InMeter / lx;
  const double sizedom2        = sizedom1 / 2;
  const double freeSpaceX      = sizedom1 * 10;
  const double freeSpaceY      = sizedom1 * 5;

  //  const double a = 100*1e-4/ lx;
  //  const double sizedom1        = 2*a ;
  //  const double sizedom2        = sizedom1 / 2;
  //  const double freeSpaceX        = 10*a;
  //  const double freeSpaceY        = 5*a;

  auto isInsidePredicate = [&](auto&& coord) {
    if (coord[0] > freeSpaceX / 2 + sizedom1 / 2 + 1e-8
        or coord[0] < freeSpaceX / 2 - sizedom1 / 2 - 1e-8)
      return false;
    else if (coord[1] > freeSpaceY / 2 + sizedom2 / 2 + 1e-8
             or coord[1] < freeSpaceY / 2 - sizedom2 / 2 - 1e-8)
      return false;
    else
      return true;
  };

  using Grid        = Dune::YaspGrid<gridDim>;
  const size_t elex = 120;
  const size_t eley = elex / 2;
  const size_t elez = 1;
  const double Lx   = freeSpaceX;
  const double Ly   = freeSpaceY;
  const double Lz   = freeSpaceY;


  Dune::FieldVector<double, gridDim> bbox;
  std::array<int, gridDim> eles{};
  if constexpr (gridDim == 2) {
    bbox = {Lx, Ly};
    eles = {elex, eley};
  } else if constexpr (gridDim == 3) {
  }

  auto grid = std::make_shared<Grid>(bbox, eles);

  //  using Grid = Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>;
  //  auto grid  = Dune::GmshReader<Grid>::read("../../examples/src/testFiles/circle.msh", false);

  grid->globalRefine(0);
  auto gridView = grid->leafGridView();

  //  draw(gridView);
  spdlog::info("The exchange length is {}.", lx);
  spdlog::info("The domain has a length of {}.", sizedom1);

  using namespace Dune::Functions::BasisFactory;
  //  auto basisEmbedded = makeBasis(gridView, power<directorDim>(lagrange<magnetizationOrder>(),
  //  BlockedInterleaved()));
  auto basisEmbeddedC
      = makeBasis(gridView, composite(power<directorDim>(lagrange<magnetizationOrder>(), BlockedInterleaved()),
                                      power<vectorPotDim>(lagrange<vectorPotOrder>(), BlockedInterleaved()),
                                      BlockedLexicographic{}));

  //  auto basisRie = makeBasis(gridView, power<directorCorrectionDim>(lagrange<magnetizationOrder>(),
  //  FlatInterleaved()));
  auto basisRieC = makeBasis(
      gridView, composite(power<directorCorrectionDim>(lagrange<magnetizationOrder>(), FlatInterleaved()),
                          power<vectorPotDim>(lagrange<vectorPotOrder>(), FlatInterleaved()), FlatLexicographic{}));
  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basisRieC.size() << " Dofs" << std::endl;

  //  draw(gridView);

  std::vector<Ikarus::FiniteElements::MicroMagneticsWithVectorPotential<decltype(basisEmbeddedC), decltype(basisRieC)>>
      fes;
  auto volumeLoad = [](auto& globalCoord, auto& lamb) {
    Eigen::Vector<double, directorDim> fext;
    fext.setZero();
    fext[0] = lamb;
    fext[1] = lamb;
    return fext;
  };

  for (auto& element : elements(gridView)) {
    auto geoCoord = element.geometry().center();
    fes.emplace_back(basisEmbeddedC, basisRieC, element, mat, volumeLoad, isInsidePredicate(geoCoord));
  }

  DirectorVector mBlocked(basisEmbeddedC.size({Dune::Indices::_0}));
  for (auto& msingle : mBlocked) {
    msingle.setValue(0.1 * Eigen::Vector<double, directorDim>::Random() + Eigen::Vector<double, directorDim>::UnitZ());
  }

  VectorPotVector aBlocked(basisEmbeddedC.size({Dune::Indices::_1}));
  for (auto& asingle : aBlocked) {
    asingle.setValue(Eigen::Vector<double, vectorPotDim>::Zero());
  }

  MultiTypeVector mAndABlocked(mBlocked, aBlocked);

  std::vector<bool> dirichletFlags(basisRieC.size(), false);
  std::cout << "dirichletFlags.size()" << dirichletFlags.size() << std::endl;
  // Fix vector potential on the whole boundary
  Dune::Functions::forEachBoundaryDOF(Dune::Functions::subspaceBasis(basisRieC, Dune::Indices::_1),
                                      [&](auto&& globalIndex) { dirichletFlags[globalIndex[0]] = true; });

  auto magnetBasis         = Dune::Functions::subspaceBasis(basisRieC, Dune::Indices::_0);
  auto localView           = magnetBasis.localView();
  auto seDOFs              = subEntityDOFs(magnetBasis);
  const auto& gridViewMagn = magnetBasis.gridView();
  for (auto&& element : elements(gridViewMagn)) {
    localView.bind(element);
    for (const auto& intersection : intersections(gridViewMagn, element))
      if (!isInsidePredicate(intersection.geometry().center())) {
        for (auto localIndex : seDOFs.bind(localView, intersection))
          dirichletFlags[localView.index(localIndex)[0]];//=true;
      }
  }

  auto magnetBasisBlocked   = Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_0);
  auto localView2           = magnetBasisBlocked.localView();
  auto seDOFs2              = subEntityDOFs(magnetBasisBlocked);
  const auto& gridViewMagn2 = magnetBasisBlocked.gridView();
  for (auto&& element : elements(gridViewMagn2)) {
    localView2.bind(element);
    for (const auto& intersection : intersections(gridViewMagn2, element))
      for (auto localIndex : seDOFs2.bind(localView2, intersection))
        if (!isInsidePredicate(intersection.geometry().center())) {
          auto b = mAndABlocked[Dune::Indices::_0][localView2.index(localIndex)[1]].begin();
          auto e = mAndABlocked[Dune::Indices::_0][localView2.index(localIndex)[1]].end();
//                    std::fill(b,e,0.0);
        }
  }

  auto denseAssembler  = DenseFlatAssembler(basisRieC, fes, dirichletFlags);
  auto sparseAssembler = SparseFlatAssembler(basisRieC, fes, dirichletFlags);
  double lambda        = 1.0;

  auto residualFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements<MultiTypeVector> req;
    req.sols.emplace_back(disp);
    req.parameter.insert({Ikarus::FEParameter::loadfactor, lambdaLocal});
    return denseAssembler.getReducedVector(req);
  };

  auto hessianFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements<MultiTypeVector> req;
    req.sols.emplace_back(disp);
    req.parameter.insert({Ikarus::FEParameter::loadfactor, lambdaLocal});
    return sparseAssembler.getReducedMatrix(req);
  };

  auto energyFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto {
    Ikarus::FErequirements<MultiTypeVector> req;
    req.sols.emplace_back(disp);
    req.parameter.insert({Ikarus::FEParameter::loadfactor, lambdaLocal});
    return denseAssembler.getScalar(req);
  };

  //  auto& h = hessianFunction(mAndABlocked, lambda);
  //    std::cout <<"hbig"<< h << std::endl;
  //
  //  auto& g = residualFunction(mAndABlocked, lambda);
  //    std::cout <<"g"<< g << std::endl;
  //
  //  auto e = energyFunction(mAndABlocked, lambda);
  //    std::cout <<"e"<< e << std::endl;

  //  assert(g.size() == gridView.size(2) * directorCorrectionDim +gridView.size(2) * vectorPotDim -
  //  std::ranges::count(dirichletFlags, true)
  //         && "The returned gradient has incorrect size");

  auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energyFunction, residualFunction, hessianFunction),
                                            parameter(mAndABlocked, lambda));
  std::cout << "CP4" << std::endl;
  auto updateFunction = std::function([&](MultiTypeVector& multiTypeVector, const Eigen::VectorXd& d) {
    auto dFull = denseAssembler.createFullVector(d);
    multiTypeVector += dFull;
  });

  checkGradient(nonLinOp, true, updateFunction);
  checkHessian(nonLinOp, true, updateFunction);

  auto nr = Ikarus::makeTrustRegion(nonLinOp, updateFunction);
  //  auto nr = Ikarus::makeTrustRegion< decltype(nonLinOp),PreConditioner::DiagonalPreconditioner>(nonLinOp,
  //  updateFunction);
  nr->setup({.verbosity = 1,
             .maxiter   = 100000,
             .grad_tol  = 1e-8,
             .corr_tol  = 1e-16,
             .useRand   = false,
             .rho_reg   = 1e6,
             .Delta0    = 1000});

  auto lc = Ikarus::LoadControl(nr, 1, {0, 100000});

  auto scalarMagnBasis          = makeBasis(gridView, lagrangeDG<magnetizationOrder>());
  auto localViewScalarMagnBasis = scalarMagnBasis.localView();

  std::vector<double> gradMNodalRes(scalarMagnBasis.size());
  std::vector<Dune::FieldVector<double,3>> curlANodalRes(scalarMagnBasis.size());

  auto writerObserver = std::make_shared<Ikarus::GenericControlObserver>(ControlMessages::STEP_ENDED, [&](auto i) {
    auto mGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, directorDim>>(
        Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_0), mAndABlocked);
    auto AGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, vectorPotDim>>(
        Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_1), mAndABlocked);

    Dune::VTKWriter vtkWriter(gridView, Dune::VTK::nonconforming);
    vtkWriter.addVertexData(mGlobalFunc, Dune::VTK::FieldInfo("m", Dune::VTK::FieldInfo::Type::vector, directorDim));
    vtkWriter.addVertexData(AGlobalFunc, Dune::VTK::FieldInfo("A", Dune::VTK::FieldInfo::Type::vector, vectorPotDim));

    Ikarus::ResultRequirements<Ikarus::FErequirements<MultiTypeVector>> resultRequirements;
    resultRequirements.req.sols.emplace_back(mAndABlocked);
    resultRequirements.req.parameter.insert({Ikarus::FEParameter::loadfactor, lambda});
    resultRequirements.resType = ResultType::gradientNormOfMagnetization;
    auto localmFunction        = localFunction(mGlobalFunc);

    auto ele = elements(gridView).begin();
    Eigen::VectorXd result;
    for (auto& fe : fes) {
      localViewScalarMagnBasis.bind(*ele);
      const auto& fe2 = localViewScalarMagnBasis.tree().finiteElement();
      const auto& referenceElement
          = Dune::ReferenceElements<double, gridDim>::general(ele->type());
      for (auto c = 0UL; c < fe2.size(); ++c) {
        const auto fineKey                        = fe2.localCoefficients().localKey(c);
        const auto nodalPositionInChildCoordinate = referenceElement.position(fineKey.subEntity(), fineKey.codim());

        auto coord = toEigenVector(nodalPositionInChildCoordinate);

        fe.calculateAt(resultRequirements, coord, result);
        gradMNodalRes[localViewScalarMagnBasis.index(localViewScalarMagnBasis.tree().localIndex(c))] = result[0];
      }
      ++ele;
    }

    auto scalarMagnBasis2          = makeBasis(gridView, power<3>(lagrangeDG<vectorPotOrder>()));
    auto localViewScalarMagnBasis2 = scalarMagnBasis2.localView();
    auto ele2 = elements(gridView).begin();
    for (auto& fe : fes) {
      localViewScalarMagnBasis2.bind(*ele2);
      const auto& fe2 = localViewScalarMagnBasis2.tree().child(0).finiteElement();
      const auto& referenceElement
          = Dune::ReferenceElements<double, gridDim>::general(ele2->type());
      for (auto c = 0UL; c < fe2.size(); ++c) {
        const auto fineKey                        = fe2.localCoefficients().localKey(c);
        const auto nodalPositionInChildCoordinate = referenceElement.position(fineKey.subEntity(), fineKey.codim());

        auto coord = toEigenVector(nodalPositionInChildCoordinate);

        resultRequirements.resType = ResultType::curlOfVectorPotential;
        fe.calculateAt(resultRequirements, coord, result);
          curlANodalRes[localViewScalarMagnBasis2.index(localViewScalarMagnBasis2.tree().child(0).localIndex(c))[0]] = Dune::FieldVector<double,3>({result[0], result[1], result[2]});

      }
      ++ele2;
    }

    auto gradmGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(scalarMagnBasis, gradMNodalRes);
    auto curlAGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double,3>>(scalarMagnBasis, curlANodalRes);

    vtkWriter.addVertexData(gradmGlobalFunc, Dune::VTK::FieldInfo("gradMNorm", Dune::VTK::FieldInfo::Type::scalar, 1));
    vtkWriter.addVertexData(curlAGlobalFunc, Dune::VTK::FieldInfo("curlA", Dune::VTK::FieldInfo::Type::vector, 3));
    auto isInsideFunc = Dune::Functions::makeAnalyticGridViewFunction(isInsidePredicate, gridView);
    vtkWriter.addCellData(isInsideFunc, Dune::VTK::FieldInfo("isInside", Dune::VTK::FieldInfo::Type::scalar, 1));
    vtkWriter.write(std::string("Magnet") + std::to_string(i));
  });
  lc.subscribeAll(writerObserver);
  lc.run();
  nonLinOp.update<0>();

  for (auto& mS : mBlocked)
    if (not Dune::FloatCmp::eq(mS.getValue().norm(), 1.0))
      std::cout << "wrong director found " << mS.getValue().transpose() << std::endl;
  //  std::cout << mBlocked;
}