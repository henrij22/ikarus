//
// Created by Alex on 21.07.2021.
//
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testHelpers.h"

#include "spdlog/spdlog.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include "ikarus/Assembler/SimpleAssemblers.h"
#include "ikarus/Controlroutines/LoadControl.h"
#include "ikarus/FEManager/DefaultFEManager.h"
#include "ikarus/FiniteElements/ElasticityFE.h"
#include "ikarus/LinearAlgebra/DirichletConditionManager.h"
#include "ikarus/utils/Observer/controlLogger.h"
#include "ikarus/utils/Observer/gridDrawerObserver.h"
#include <ikarus/FiniteElements/ForceLoad.h>
#include <ikarus/Grids/GridHelper/griddrawer.h>
#include <ikarus/Grids/SimpleGrid/SimpleGrid.h>
#include <ikarus/LinearAlgebra/NonLinearOperator.h>
#include <ikarus/Solver/LinearSolver/LinearSolver.h>


TEST(LinearSolverTest, LinearSolverTest1) {
  using namespace Ikarus::Grid;
  using namespace Ikarus;
  using Grid = SimpleGrid<2, 2>;
  SimpleGridFactory<2, 2> gridFactory;
  using vertexType = Eigen::Vector2d;
  std::vector<vertexType> verticesVec;
  const double L    = 10;
  const double h    = 1;
  const size_t elex = 5;
  const size_t eley = 5;
  for (size_t j = 0; j < eley + 1; ++j) {
    for (size_t i = 0; i < elex + 1; ++i)
      verticesVec.emplace_back(vertexType{i * L / (elex), j * h / (eley)});
  }
  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);
  for (size_t i = 0; i < elex; ++i) {
    for (size_t j = 0; j < eley; ++j) {
      std::vector<size_t> elementIndices;
      elementIndices.resize(4);
      elementIndices
          = {i + j * (elex + 1), i + j * (elex + 1) + 1, i + (j + 1) * (elex + 1), i + (j + 1) * (elex + 1) + 1};
      gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
    }
  }

  Grid grid     = gridFactory.createGrid();
  auto gridView = grid.leafGridView();

  //  draw(gridView);

  std::vector<Ikarus::FiniteElements::IFiniteElement> feContainer;

  for (auto&& ge : rootEntities(gridView))
    feContainer.emplace_back(Ikarus::FiniteElements::ElasticityFE(ge, gridView.indexSet(), 1000, 0.0));

  auto feManager = Ikarus::FEManager::DefaultFEManager(feContainer, gridView);

  Ikarus::DirichletConditionManager dirichletConditionManager(feManager);
  dirichletConditionManager.addConstraint(vertices(gridView).front(), 0);
  dirichletConditionManager.addConstraint(vertices(gridView).back(), 1);
  dirichletConditionManager.addConstraint(vertices(gridView).at(3), 1);
  dirichletConditionManager.finalize();

  auto vectorAssembler = Ikarus::Assembler::VectorAssembler(feManager, dirichletConditionManager);

  auto denseMatrixAssembler  = Ikarus::Assembler::DenseMatrixAssembler(feManager, dirichletConditionManager);
  auto sparseMatrixAssembler = Ikarus::Assembler::SparseMatrixAssembler(feManager,dirichletConditionManager);

  auto& x = feManager.getVariables();

  Ikarus::ILinearSolver<double> solver(SolverTypeTag::LDLT);
  auto& b = vectorAssembler.getReducedVector(FiniteElements::forces);
  b[3]=1;
  auto& A = denseMatrixAssembler.getReducedMatrix(FiniteElements::stiffness);
  solver.compute(A);
  auto sol = solver.solve(b);
  std::cout<<sol.transpose()<<std::endl;

  auto& Asparse = sparseMatrixAssembler.getMatrix(FiniteElements::stiffness);
  Ikarus::ILinearSolver<double> solverCG(SolverTypeTag::ConjugateGradient);
  solverCG.compute(Asparse);
  b = vectorAssembler.getVector(FiniteElements::forces);
  auto sol2 = solverCG.solve(b);
  std::cout<<sol2.transpose()<<std::endl;
  EXPECT_THROW(solver.compute(Asparse),std::logic_error);

  Ikarus::ILinearSolver<double> solver3(SolverTypeTag::CholmodSupernodalLLT);
  solver3.compute(Asparse);
  auto sol3 = solverCG.solve(b);
  std::cout<<sol3.transpose()<<std::endl;
  EXPECT_THAT(sol2, EigenApproxEqual(sol3, 1e-14));
}