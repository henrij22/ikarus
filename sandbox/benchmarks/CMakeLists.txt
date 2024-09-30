# SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)


add_executable("Benchmark" benchmark.cpp)
set_target_properties("Benchmark" PROPERTIES CXX_STANDARD 20 OUTPUT_NAME "benchmark")
add_dune_pythonlibs_flags("Benchmark")
target_compile_options(
  "Benchmark" PUBLIC -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
                        -Wno-deprecated-copy -ftemplate-backtrace-limit=0
)

target_link_libraries("Benchmark" PUBLIC ikarus)
add_dependencies(AllExamples "Benchmark")
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries("Benchmark" PUBLIC ${PYTHON_LIBRARIES})


# Copy the example grid used for testing into the build dir
#file(COPY auxiliaryFiles/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/auxiliaryFiles)
