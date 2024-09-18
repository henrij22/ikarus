// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * \file nonlinearenhancedassumedstrains.hh
 * \brief Definition of the nonlinear EAS class.
 * \ingroup  mechanics
 */

#pragma once
#if HAVE_DUNE_LOCALFEFUNCTIONS
  #include <dune/localfefunctions/derivativetransformators.hh>
  #include <dune/localfefunctions/linearAlgebraHelper.hh>
  #include <dune/localfefunctions/meta.hh>

  #include <ikarus/finiteelements/ferequirements.hh>
  #include <ikarus/finiteelements/mechanics/easvariants.hh>

namespace Ikarus {

/**
 * \brief Wrapper class for using Enhanced Assumed Strains (EAS) with displacement based elements.
 *
 * \ingroup mechanics
 *
 * This class extends a displacement-based element to support Enhanced Assumed Strains.
 *
 * \tparam DFE The base displacement-based element type.
 */
template <typename DFE>
class NonlinearEnhancedAssumedStrains : public DFE
{
public:
  using DisplacementBasedElement = DFE;
  using FERequirementType        = typename DisplacementBasedElement::FERequirementType;
  using LocalView                = typename DisplacementBasedElement::LocalView;
  using Geometry                 = typename DisplacementBasedElement::Geometry;
  using GridView                 = typename DisplacementBasedElement::GridView;
  using Traits                   = typename DisplacementBasedElement::Traits;
  using DisplacementBasedElement::localView;

  /**
* \brief Constructor for Enhanced Assumed Strains elements.
  * \details Disabling this forwarding constructor if the argument provided is NonlinearEnhancedAssumedStrains itself,
to forward the
  // calls to the implicit copy constructor
* \tparam Args Variadic template for constructor arguments.
* \param args Constructor arguments forwarded to the base class.
*/
  template <typename... Args>
  requires(not std::is_same_v<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Args...>>>,
                              NonlinearEnhancedAssumedStrains>)
  explicit NonlinearEnhancedAssumedStrains(Args&&... args)
      : DisplacementBasedElement(std::forward<Args>(args)...) {}

  /**
   * \brief Calculates a scalar quantity for the element.
   *
   * This function calculates a scalar quantity for the element based on the FERequirementType.
   *
   * \param par The FERequirementType object.
   * \return Computed scalar quantity.
   */
  double calculateScalar(const FERequirementType& par) const {
    if (isDisplacementBased())
      return DisplacementBasedElement::calculateScalar(par);
    DUNE_THROW(Dune::NotImplemented,
               "EAS element do not support any scalar calculations, i.e. they are not derivable from a potential");
  }

  /**
   * \brief Checks if the element is displacement-based and the EAS is turned off.
   *
   * \return True if the element is displacement-based, false otherwise.
   */
  bool isDisplacementBased() const { return std::holds_alternative<std::monostate>(easVariant_); }

  /**
   * \brief Calculates vectorial quantities for the element.
   *
   * This function calculates the vectorial quantities for the element based on the FERequirementType.
   *
   * \param par The FERequirementType object.
   * \param force Vector to store the calculated forces.
   */
  inline void calculateVector(const FERequirementType& par, typename Traits::template VectorType<> force) const {
    calculateVectorImpl<double>(par, force);
  }

  /**
   * \brief Gets the variant representing the type of Enhanced Assumed Strains (EAS).
   *
   * \return Const reference to the EAS variant.
   */
  const auto& easVariant() const { return easVariant_; }

  /**
   * \brief Gets the number of EAS parameters based on the current EAS type.
   *
   * \return Number of EAS parameters.
   */
  auto getNumberOfEASParameters() const {
    return std::visit(
        [&]<typename EAST>(const EAST&) {
          if constexpr (std::is_same_v<std::monostate, EAST>)
            return 0;
          else
            return EAST::enhancedStrainSize;
        },
        easVariant_);
  }

  /**
   * \brief Initializes the internal parameter alpha_ based on the number of EAS parameters.
   */
  void initializeIteration() {
    if (isDisplacementBased())
      return;
    alpha_.resize(getNumberOfEASParameters());
    alpha_.setZero();
  }

  /**
   * \brief Updates the internal parameter alpha_ at the end of an iteration
   * when NonLinearSolverMessages::SOLUTION_UPDATED is notified ny the non-linear solver.
   *
   * \param par The FERequirementType object.
   */
  void finalizeIteration(const FERequirementType& par, const Eigen::VectorXd& correction_) {
    if (isDisplacementBased())
      return;
    const auto& Rtilde      = calculateRtilde(par);
    const auto localdxBlock = Ikarus::FEHelper::localSolutionBlockVector<Traits, double>(correction_, localView());
    const auto localdx      = Dune::viewAsFlatEigenVector(localdxBlock);

    std::visit(
        [&]<typename EAST>(const EAST& easFunction) {
          if constexpr (not std::is_same_v<std::monostate, EAST>) {
            constexpr int enhancedStrainSize = EAST::enhancedStrainSize;
            Eigen::Matrix<double, enhancedStrainSize, enhancedStrainSize> D;
            calculateDAndLMatrix(easFunction, par, D, L_);
            const decltype(alpha_) updateAlpha = D.inverse() * (Rtilde + (L_ * localdx).eval());
            this->alpha_ -= updateAlpha;
          }
        },
        easVariant_);
  }

  const auto& alpha() const { return alpha_; }

  /**
   * \brief Calculates the matrix for the element.
   *
   * This function calculates the matrix for the element based on the FERequirementType.
   *
   * \param par The FERequirementType object.
   * \param K Matrix to store the calculated stiffness.
   */
  void calculateMatrix(const FERequirementType& par, typename Traits::template MatrixType<> K) const {
    using namespace Dune::DerivativeDirections;
    using namespace Dune;

    /// fill K with displacement-based stiffness.
    /// It is assumed to be assembled block-wise on element level.
    /// This means the displacements x,y,z of node I are grouped together.
    DisplacementBasedElement::calculateMatrix(par, K);

    if (isDisplacementBased())
      return;

    const auto geo            = localView().element().geometry();
    const auto& numberOfNodes = DisplacementBasedElement::numberOfNodes();
    auto strainFunction       = DisplacementBasedElement::strainFunction(par);
    const auto uFunction      = DisplacementBasedElement::displacementFunction(par);
    const auto disp           = Dune::viewAsFlatEigenVector(uFunction.coefficientsRef());
    std::visit(
        [&]<typename EAST>(const EAST& easFunction) {
          if constexpr (not std::is_same_v<std::monostate, EAST>) {
            constexpr int enhancedStrainSize = EAST::enhancedStrainSize;
            Eigen::Matrix<double, enhancedStrainSize, enhancedStrainSize> D;
            calculateDAndLMatrix(easFunction, par, D, L_);
            for (const auto& [gpIndex, gp] : strainFunction.viewOverIntegrationPoints()) {
              const auto M            = easFunction.calcM(gp.position());
              const double intElement = geo.integrationElement(gp.position()) * gp.weight();
              const auto EVoigt       = strainFunction.evaluate(gpIndex, on(gridElement)).eval();
              const auto CEval        = DisplacementBasedElement::materialTangent(EVoigt).eval();
              auto stresses           = (CEval * M * alpha_).eval();
              for (size_t i = 0; i < numberOfNodes; ++i) {
                for (size_t j = 0; j < numberOfNodes; ++j) {
                  const auto kgIJ =
                      strainFunction.evaluateDerivative(gpIndex, wrt(coeff(i, j)), along(stresses), on(gridElement));
                  K.template block<Traits::mydim, Traits::mydim>(i * Traits::mydim, j * Traits::mydim) +=
                      kgIJ * intElement;
                }
              }
            }
            K -= L_.transpose() * D.inverse() * L_;
          }
        },
        easVariant_);
  }

  /**
   * \brief Calculates a requested result at a specific local position using the Enhanced Assumed Strains (EAS)
   * method.
   *
   * This function calculates the results at the specified local coordinates .
   * It takes into account the displacement-based element calculations and, if applicable, incorporates the EAS method
   * for enhanced accuracy.
   *
   * \param req The result requirements.
   * \param local The local coordinates at which results are to be calculated.
   * \return calculated result
   *
   * \tparam resType The type representing the requested result.
   */
  template <ResultType resType>
  auto calculateAt(const FERequirementType& req, const Dune::FieldVector<double, Traits::mydim>& local) const {
    DUNE_THROW(Dune::NotImplemented, "calculateAt not implemented for Nonlinear EAS.");
  }

  /**
   * \brief Sets the EAS type for 2D elements.
   *
   * \param numberOfEASParameters The number of EAS parameters
   */
  void setEASType(int numberOfEASParameters) {
    const auto& numberOfNodes = DisplacementBasedElement::numberOfNodes();
    if (not((numberOfNodes == 4 and Traits::mydim == 2) or (numberOfNodes == 8 and Traits::mydim == 3)) and
        (not isDisplacementBased()))
      DUNE_THROW(Dune::NotImplemented, "EAS only supported for Q1 or H1 elements");

    if constexpr (Traits::mydim == 2) {
      switch (numberOfEASParameters) {
        case 0:
          easVariant_ = std::monostate();
          break;
        case 4:
          easVariant_ = EAS::Q1E4(localView().element().geometry());
          break;
        case 5:
          easVariant_ = EAS::Q1E5(localView().element().geometry());
          break;
        case 7:
          easVariant_ = EAS::Q1E7(localView().element().geometry());
          break;
        default:
          DUNE_THROW(Dune::NotImplemented, "The given EAS parameters are not available for the 2D case.");
          break;
      }
    } else if constexpr (Traits::mydim == 3) {
      switch (numberOfEASParameters) {
        case 0:
          easVariant_ = std::monostate();
          break;
        case 9:
          easVariant_ = EAS::H1E9(localView().element().geometry());
          break;
        case 21:
          easVariant_ = EAS::H1E21(localView().element().geometry());
          break;
        default:
          DUNE_THROW(Dune::NotImplemented, "The given EAS parameters are not available for the 3D case.");
          break;
      }
    }
  }

protected:
  template <typename ScalarType>
  inline ScalarType calculateScalarImpl(
      const FERequirementType& par, const std::optional<const Eigen::VectorX<ScalarType>>& dx = std::nullopt) const {
    if (isDisplacementBased())
      return DisplacementBasedElement::template calculateScalarImpl<ScalarType>(par, dx);
    DUNE_THROW(Dune::NotImplemented,
               "EAS element do not support any scalar calculations, i.e. they are not derivable from a potential");
  }

  template <typename ScalarType>
  void calculateVectorImpl(const FERequirementType& par, typename Traits::template VectorType<ScalarType> force,
                           const std::optional<const Eigen::VectorX<ScalarType>>& dx = std::nullopt) const {
    DisplacementBasedElement::calculateVectorImpl(par, force, dx);
    if (isDisplacementBased())
      return;
    using namespace Dune;
    const auto uFunction      = DisplacementBasedElement::displacementFunction(par, dx);
    auto strainFunction       = DisplacementBasedElement::strainFunction(par, dx);
    const auto& numberOfNodes = DisplacementBasedElement::numberOfNodes();
    const auto disp           = Dune::viewAsFlatEigenVector(uFunction.coefficientsRef());

    using namespace Dune::DerivativeDirections;

    const auto geo = localView().element().geometry();

    // Internal forces from enhanced strains
    std::visit(
        [&]<typename EAST>(const EAST& easFunction) {
          if constexpr (not std::is_same_v<std::monostate, EAST>) {
            constexpr int enhancedStrainSize = EAST::enhancedStrainSize;
            Eigen::Matrix<double, enhancedStrainSize, enhancedStrainSize> D;
            calculateDAndLMatrix(easFunction, par, D, L_);
            const auto& Rtilde = calculateRtilde(par, dx);
            for (const auto& [gpIndex, gp] : strainFunction.viewOverIntegrationPoints()) {
              const auto M            = easFunction.calcM(gp.position());
              const double intElement = geo.integrationElement(gp.position()) * gp.weight();
              const auto EVoigt       = (strainFunction.evaluate(gpIndex, on(gridElement))).eval();
              const auto CEval        = DisplacementBasedElement::materialTangent(EVoigt).eval();
              auto stresses           = (CEval * M * alpha_).eval();
              for (size_t i = 0; i < numberOfNodes; ++i) {
                const auto bopI = strainFunction.evaluateDerivative(gpIndex, wrt(coeff(i)), on(gridElement));
                force.template segment<Traits::worlddim>(Traits::worlddim * i) +=
                    bopI.transpose() * stresses * intElement;
              }
            }

            force -= L_.transpose() * D.inverse() * Rtilde;
          }
        },
        easVariant_);
  }

private:
  using EASVariant = EAS::Variants<Geometry>::type;
  EASVariant easVariant_;
  mutable Eigen::MatrixXd L_;
  Eigen::VectorXd alpha_;
  template <int enhancedStrainSize>
  void calculateDAndLMatrix(const auto& easFunction, const auto& par,
                            Eigen::Matrix<double, enhancedStrainSize, enhancedStrainSize>& DMat,
                            Eigen::MatrixXd& LMat) const {
    using namespace Dune;
    using namespace Dune::DerivativeDirections;

    auto strainFunction      = DisplacementBasedElement::strainFunction(par);
    const auto geo           = localView().element().geometry();
    const auto numberOfNodes = DisplacementBasedElement::numberOfNodes();
    DMat.setZero();
    LMat.setZero(enhancedStrainSize, localView().size());
    for (const auto& [gpIndex, gp] : strainFunction.viewOverIntegrationPoints()) {
      const auto M      = easFunction.calcM(gp.position());
      const auto EVoigt = strainFunction.evaluate(gpIndex, on(gridElement)).eval();
      const auto CEval  = DisplacementBasedElement::materialTangent(EVoigt).eval();
      const double detJ = geo.integrationElement(gp.position());
      DMat += M.transpose() * CEval * M * detJ * gp.weight();
      for (size_t i = 0U; i < numberOfNodes; ++i) {
        const size_t I = Traits::worlddim * i;
        const auto Bi  = strainFunction.evaluateDerivative(gpIndex, wrt(coeff(i)), on(gridElement));
        LMat.template block<enhancedStrainSize, Traits::worlddim>(0, I) +=
            M.transpose() * CEval * Bi * detJ * gp.weight();
      }
    }
  }

  template <typename ScalarType = double>
  Eigen::VectorX<ScalarType> calculateRtilde(
      const FERequirementType& par, const std::optional<const Eigen::VectorX<ScalarType>>& dx = std::nullopt) const {
    using namespace Dune;
    using namespace Dune::DerivativeDirections;
    const auto geo      = localView().element().geometry();
    auto strainFunction = DisplacementBasedElement::strainFunction(par, dx);
    Eigen::VectorX<ScalarType> Rtilde;
    Rtilde.resize(getNumberOfEASParameters());
    Rtilde.setZero();

    std::visit(
        [&]<typename EAST>(const EAST& easFunction) {
          if constexpr (not std::is_same_v<std::monostate, EAST>) {
            for (const auto& [gpIndex, gp] : strainFunction.viewOverIntegrationPoints()) {
              const auto M            = easFunction.calcM(gp.position());
              const double intElement = geo.integrationElement(gp.position()) * gp.weight();
              const auto EVoigt       = (strainFunction.evaluate(gpIndex, on(gridElement))).eval();
              const auto CEval        = DisplacementBasedElement::materialTangent(EVoigt).eval();
              auto stresses           = (CEval * M * alpha_).eval();
              stresses += DisplacementBasedElement::getStress(EVoigt).eval();
              Rtilde += (M.transpose() * stresses).eval() * intElement;
            }
          }
        },
        easVariant_);
    return Rtilde;
  }
};
} // namespace Ikarus

#else
  #error NonlinearEnhancedAssumedStrains depends on dune-localfefunctions, which is not included
#endif
