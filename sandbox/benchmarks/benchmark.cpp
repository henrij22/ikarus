// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/grid/yaspgrid.hh>

#include "ikarus/finiteelements/mechanics/enhancedassumedstrains.hh"
#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/controlroutines/loadcontrol.hh>
#include "ikarus/finiteelements/mechanics/materials/vanishingstress.hh"
#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/mechanics/easvariants.hh>
#include <ikarus/finiteelements/mechanics/loads.hh>
#include <ikarus/finiteelements/mechanics/materials/svk.hh>
#include <ikarus/finiteelements/mechanics/materials/vanishingstrain.hh>
#include <ikarus/finiteelements/mechanics/nonlinearelastic.hh>
#include "ikarus/io/vtkwriter.hh"
#include "ikarus/solver/nonlinearsolver/nonlinearsolverfactory.hh"
#include "ikarus/utils/dirichletvalues.hh"
#include "ikarus/utils/observer/controlvtkwriter.hh"
#include "ikarus/utils/observer/nonlinearsolverlogger.hh"
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/tensorutils.hh>

using namespace Ikarus;

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);

  auto mat =
      Ikarus::StVenantKirchhoff{toLamesFirstParameterAndShearModulus(YoungsModulusAndPoissonsRatio{1000.0, 0.3})};
  auto psMat = Ikarus::planeStress(mat);
  std::cout << psMat.materialParametersImpl().lambda << " " << psMat.materialParametersImpl().mu << std::endl;

  auto grid = Dune::YaspGrid<2>{
      Dune::FieldVector<double, 2>{ 1, 0.1},
       std::array<int, 2>{10,   1}
  };
  // grid.globalRefine(3);
  auto gridView = grid.leafGridView();

  using namespace Dune::Functions::BasisFactory;
  auto basis = Ikarus::makeBasis(gridView, power<2>(lagrange<1>()));

  Dune::BitSetVector<1> neumannVertices(gridView.size(2), false);

  auto& indexSet = gridView.indexSet();
  for (auto&& vertex : vertices(gridView)) {
    neumannVertices[indexSet.index(vertex)] = Dune::FloatCmp::eq(vertex.geometry().center()[0], 1.0);
  }

  BoundaryPatch<decltype(gridView)> neumannBoundary(gridView, neumannVertices);

  auto load = [](auto& coords, auto& lambda) { return Eigen::Vector2d{0, -0.1 * lambda}; };

  auto preFE = Ikarus::nonLinearElastic(psMat);
  auto sk    = Ikarus::skills(preFE, eas(4), Ikarus::neumannBoundaryLoad(&neumannBoundary, load));

  using FEType = decltype(makeFE(basis, sk));

  std::vector<FEType> fes;
  for (auto&& ge : elements(gridView)) {
    fes.emplace_back(makeFE(basis, sk));
    fes.back().bind(ge);
  }

  Ikarus::DirichletValues dirichletValues(basis.flat());
  dirichletValues.fixBoundaryDOFs([&](auto& dirichletFlags, auto&& localIndex, auto&& localView, auto&& intersection) {
    if (std::abs(intersection.geometry().center()[0]) < 1e-8)
      dirichletFlags[localView.index(localIndex)] = true;
  });

  Eigen::VectorXd d;
  d.setZero(basis.flat().size());
  double lambda = 0.0;

  auto req = typename FEType::Requirement();
  req.insertGlobalSolution(d).insertParameter(lambda);

  auto affo = AffordanceCollection(ScalarAffordance::noAffordance, VectorAffordance::forces, MatrixAffordance::stiffness);
  auto sparseAssembler = makeSparseFlatAssembler(fes, dirichletValues);
  
  sparseAssembler->bind(req, affo, Ikarus::DBCOption::Full);

  auto linSolver = LinearSolver{SolverTypeTag::sd_SparseLU};
  auto config    = NewtonRaphsonConfig<decltype(linSolver)>{
         .parameters = {.tol = 1e-8, .maxIter = 100},
           .linearSolver = linSolver
  };
  auto nonLinSolver = NonlinearSolverFactory{config}.create(sparseAssembler);

  // auto info = nonLinSolver->solve();

  auto lc = LoadControl{
      nonLinSolver, 10, {0, 20}
  };

  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();

  auto vtkWriter = std::make_shared<ControlSubsamplingVertexVTKWriter<std::remove_cvref_t<decltype(basis.flat())>>>(
      basis.flat(), d, 2);

  vtkWriter->setFileNamePrefix("series");
  vtkWriter->setFieldInfo("Displacement", Dune::VTK::FieldInfo::Type::vector, 2);

  lc.nonlinearSolver().subscribeAll(nonLinearSolverObserver);
  lc.subscribeAll(vtkWriter);

  auto info = lc.run();

  using Ikarus::Vtk::DataTag::asPointData;
  Ikarus::Vtk::Writer writer(sparseAssembler);

  writer.template addResult<Ikarus::ResultTypes::PK2Stress>(asPointData);
  writer.addInterpolation(d, basis.flat(), "displacement", asPointData);

  writer.write("result");

  std::cout << std::boolalpha << info.success << std::endl;
  std::cout << "Lambda " << lambda << std::endl;

  return !info.success;
}
