//
// Created by Alex on 21.04.2021.
//
#define EIGEN_MATRIXBASE_PLUGIN "IBB_Eigen_MatrixBaseAddon.h"

#include <fstream>
#include <vector>
//#include "Geometries/Geometries.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

//#include "spdlog/fmt/ostr.h"
//#include "spdlog/spdlog.h"
//#include "spdlog/sinks/basic_file_sink.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

// void foo() {
//  spdlog::info("Does this appear in the correct logger?");
//  spdlog::debug("This is a debug statement");
//}

// TEST(Dependencies, spdlog) {
//
//  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
//  console_sink->set_level(spdlog::level::trace);
//  console_sink->set_pattern("[%^%l%$] %v");
//
//  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/testLog.txt", true);
//  file_sink->set_level(spdlog::level::trace);
//  file_sink->set_pattern("[%^%l%$] %v");
//
////    spdlog::logger logger("multi_sink", {console_sink, file_sink});
//  spdlog::logger logger("multi_sink", {file_sink});
//  spdlog::set_default_logger(std::make_shared<spdlog::logger>(logger));
//
//  logger.info("Welcome to spdlog!");
//  logger.error("Some error message with arg: {}", 1);
//
//  logger.warn("Easy padding in numbers like {:08d}", 12);
//  logger.critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
//  logger.info("Support for floats {:03.2f}", 1.23456);
//  logger.info("Positional args are {1} {0}..", "too", "supported");
//  logger.info("{:<29}{}", "left aligned", "w");
//  logger.info("{:>30}", "right aligned");
//
//  logger.set_level(spdlog::level::debug); // Set global log level to debug
//  logger.debug("This message should be displayed..");
//
//  foo();
//  Eigen::Vector3d a({1.0, 2.0, 3.0});
//  logger.info("{}", a.transpose());
//  logger.flush();
//  std::ifstream t(file_sink->filename());
//  std::string file((std::istreambuf_iterator<char>(t)),
//                   (std::istreambuf_iterator<char>()));
//  std::string expectedOutput = R"xxx([info] Welcome to spdlog!
//[error] Some error message with arg: 1
//[warning] Easy padding in numbers like 00000012
//[critical] Support for int: 42;  hex: 2a;  oct: 52; bin: 101010
//[info] Support for floats 1.23
//[info] Positional args are supported too..
//[info] left aligned                 w
//[info]                  right aligned
//[debug] This message should be displayed..
//[info] Does this appear in the correct logger?
//[info] 1 2 3
//)xxx";
//  EXPECT_EQ(expectedOutput, file);
//}

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

TEST(Dependencies, dunecommonInputParser) {
  Dune::ParameterTree parameterSet;

  std::string testInPutFile = R"xxx(tolerance = 1e-12

integrationType = Gauss

[elementParameters]
thickness = 0.6

[materialParameters]
mu = 2.7191e+4
lambda = 4.4364e+4

Emod = 1000
nu = 0.3

[]
)xxx";

  std::string testInputFileName = "TestInputFile.parset";
  std::ofstream out(testInputFileName);
  out << testInPutFile;
  out.close();

  Dune::ParameterTreeParser::readINITree(testInputFileName, parameterSet);

  const auto tolerance = parameterSet.get<double>("tolerance");
  const auto integrationType = parameterSet.get<std::string>("integrationType");

  const Dune::ParameterTree &materialParameters = parameterSet.sub("materialParameters");
  const Dune::ParameterTree &elementParameters = parameterSet.sub("elementParameters");

  const auto thickness = elementParameters.get<double>("thickness");
  const auto mu = materialParameters.get<double>("mu");
  const auto lambda = materialParameters.get<double>("lambda");
  const auto Emod = materialParameters.get<double>("Emod");
  const auto nu = materialParameters.get<double>("nu");

  EXPECT_EQ(tolerance, 1e-12);
  EXPECT_EQ(mu, 2.7191e+4);
  EXPECT_EQ(lambda, 4.4364e+4);
  EXPECT_EQ(thickness, 0.6);
  EXPECT_EQ(nu, 0.3);
  EXPECT_EQ(Emod, 1000);
  EXPECT_EQ(integrationType, "Gauss");
}

////#include <dune/common/fvector.hh>
//#include <dune/grid/onedgrid.hh>
#include <cstdlib>
float foo(float f,float g) { return (f / g); }

TEST(Dependencies, dunegridEntities) {
  //  using namespace Dune;
  //  FieldVector<double,1> a = -2.0;
  //  FieldVector<double,1> b =  3.0;
  //  GridFactory<OneDGrid> gridFactory;
  //  gridFactory.insertVertex(a);
  //  gridFactory.insertVertex(b);
  //  std::vector<unsigned int> verticesIndex ({0,1});
  //
  //  gridFactory.insertElement(GeometryTypes::line,verticesIndex);
  //
  //  std::unique_ptr<OneDGrid> grid = gridFactory.createGrid();
  //  auto gridView = grid->leafGridView();
  //
  //  for (const auto& ele:elements(gridView)) {
  ////      ele.geometry();
  //  }
  // Does not work due to bug in GCC 11.1
//  foo(1.0f,0.0f);
  [[maybe_unused]] int *i = (int*) malloc(sizeof(int));
}