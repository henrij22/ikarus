//
// Created by Alex on 25.05.2021.
//

#include "VariableDefinitions.h"

#include <ikarus/Variables/InterfaceVariable.h>

namespace Ikarus::Variable {

  int valueSize(const IVariable& vo) { return vo.variableImpl->do_valueSize(); }
  int correctionSize(const IVariable& vo) { return vo.variableImpl->do_correctionSize(); }

  IVariable& operator+=(IVariable& vo, const IVariable::UpdateType& correction) {
    vo.variableImpl->do_assignAdd(correction);
    return vo;
  }

  IVariable& operator+=(IVariable* vo, const IVariable::UpdateType& correction) {
    return ((*vo) += correction);
  }

  IVariable operator+(IVariable& vo, const IVariable::UpdateType& correction) {
    IVariable res{vo};
    res.variableImpl->do_assignAdd(correction);
    return res;
  }

  IVariable operator+(IVariable* vo, const IVariable::UpdateType& correction) {
    return ((*vo) + correction);
  }

  void setValue(IVariable& vo, const IVariable::UpdateType& value) {
    return vo.variableImpl->do_setValue(value);
  }
  Ikarus::DynVectord getValue(const IVariable& vo) { return vo.variableImpl->do_getValue(); }
  size_t getTag(const IVariable& var) { return var.variableImpl->do_getTag(); }
  bool operator==(const IVariable& var, const IVariable& other) {
    return var.variableImpl->do_equalComparison(other);
  }
  bool operator<(const IVariable& var, const IVariable& other) {
    return var.variableImpl->do_lessComparison(other);
  }

  std::ostream& operator<<(std::ostream& s, const IVariable& var) {
    s << var.variableImpl->do_getValue().transpose() << '\n' << " Tag: " << getName(var)<< '\n';
    return s;
  }
  size_t correctionSize(std::span<const IVariable> varSpan) {
    return std::accumulate(varSpan.begin(), varSpan.end(), 0,
                           [](size_t cursize, const IVariable& var) { return cursize + correctionSize(var); });
  }
  void update(std::span<IVariable> varSpan, const Ikarus::DynVectord& correction) {
    assert(static_cast<Eigen::Index>(correctionSize(varSpan)) == correction.size());
    // update Variable
    Eigen::Index posHelper = 0;
    std::for_each(varSpan.begin(), varSpan.end(), [&](IVariable& var) {
      var += correction.segment(posHelper, correctionSize(var));
      posHelper += correctionSize(var);
    });
  }

  size_t valueSize(std::span<const IVariable> varSpan) {
    return std::accumulate(varSpan.begin(), varSpan.end(), 0,
                           [](size_t cursize, const IVariable& var) { return cursize + valueSize(var); });
  }

  std::string getName(const IVariable&var){
    return Ikarus::Variable::variableNames [getTag(var)];
  }

}  // namespace Ikarus::Variable