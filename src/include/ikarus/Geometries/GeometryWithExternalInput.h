//
// Created by Alex on 21.04.2021.
//

#pragma once

#include <concepts>
#include <iostream>
#include <span>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "ikarus/Interpolators/Interpolator.h"
#include <ikarus/utils/LinearAlgebraHelper.h>

namespace Ikarus::Geometry {

  template <typename ct, int wdim, int geodim>
    requires requires { geodim <= wdim; }
  class GeometryWithExternalInput {
  public:
    /** \brief Type used for coordinates */
    using ctype = ct;

    /** \brief Dimension of the world space */
    static constexpr int coorddimension = wdim;

    /** \brief Dimension of the geometry */
    static constexpr int mydimension = geodim;

    /** \brief Type for local coordinate vector */
    using LocalCoordinate = Eigen::Matrix<ctype, mydimension, 1>;

    /** \brief Type for coordinate vector in world space */
    using GlobalCoordinate = Eigen::Matrix<ctype, coorddimension, 1>;

    /** \brief Type for the transposed Jacobian matrix */
    using JacobianTransposed = Eigen::Matrix<ctype, mydimension, coorddimension>;

    /** \brief Type for the transposed inverse Jacobian matrix */
    using JacobianInverseTransposed = Eigen::Matrix<ctype, coorddimension, mydimension>;

    template <typename DerivedAnsatzFunctionType, typename GlobalCoordinateListType>
    static ctype determinantJacobian(const Eigen::MatrixBase<DerivedAnsatzFunctionType>& dN,
                                     const Eigen::MatrixBase<GlobalCoordinateListType>& nodevalueList) {
      const auto JT = jacobianTransposed(dN, nodevalueList);
      return sqrt((JT * JT.transpose()).determinant());
    }

    template <typename DerivedAnsatzFunctionType, typename GlobalCoordinateListType>
    static JacobianTransposed jacobianTransposed(const Eigen::MatrixBase<DerivedAnsatzFunctionType>& dN,
                                                 const Eigen::MatrixBase<GlobalCoordinateListType>& nodevalueList) {
      static_assert(DerivedAnsatzFunctionType::ColsAtCompileTime == mydimension);
      static_assert(GlobalCoordinateListType::RowsAtCompileTime == coorddimension);
      assert(dN.rows() == nodevalueList.cols());
      JacobianTransposed JT;
      for (int i = 0; i < JT.rows(); ++i)
        JT.row(i) = interpolate(dN.col(i), nodevalueList).transpose();

      return JT;
    }

    template <typename DerivedAnsatzFunctionType, typename GlobalCoordinateListType>
    static JacobianInverseTransposed jacobianInverseTransposed(
        const Eigen::MatrixBase<DerivedAnsatzFunctionType>& dN,
        const Eigen::MatrixBase<GlobalCoordinateListType>& nodevalueList) {
      return jacobianTransposed(dN, nodevalueList).completeOrthogonalDecomposition().pseudoInverse();
    }

    template <typename DerivedAnsatzFunctionType, typename GlobalCoordinateListType>
    static DerivedAnsatzFunctionType transformCurvLinearDerivativesToCartesian(
        const Eigen::MatrixBase<DerivedAnsatzFunctionType>& dN,
        const Eigen::MatrixBase<GlobalCoordinateListType>& nodevalueList) {
      const JacobianTransposed jT = jacobianTransposed(dN, nodevalueList);
      const auto jCart            = Ikarus::LinearAlgebra::orthonormalizeMatrixColumns(jT.transpose());

      return dN * (jT * jCart).inverse().transpose();
    }
  };

}  // namespace Ikarus::Geometry
