// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>
#include <dune/grid/yaspgrid.hh>

#include <iostream>

#include <ikarus/utils/init.hh>

int main(int argc, char** argv) {
   Ikarus::init(argc, argv); 

  auto grid = Dune::YaspGrid<2>{Dune::FieldVector<double, 2>{1.0, 0.1}, std::array<int, 2>{10, 1}};
  
  
}
