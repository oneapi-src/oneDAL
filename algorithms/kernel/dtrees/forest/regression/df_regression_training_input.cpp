/* file: df_regression_training_input.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training { Status checkImpl(const decision_forest::training::Parameter& prm); }

namespace regression
{
namespace training
{
namespace interface1
{

Parameter::Parameter(){}
Status Parameter::check() const
{
    return decision_forest::training::checkImpl(*this);
}

/** Default constructor */
Input::Input() : algorithms::regression::training::Input(lastInputId + 1) {}

/**
 * Returns an input object for decision forest model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return algorithms::regression::training::Input::get(algorithms::regression::training::InputId(id));
}

/**
 * Sets an input object for decision forest model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    algorithms::regression::training::Input::set(algorithms::regression::training::InputId(id), value);
}

/**
* Checks an input object for the decision forest algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/

Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::training::Input::check(par, method));
    NumericTablePtr dataTable = get(data);
    NumericTablePtr dependentVariableTable = get(dependentVariable);

    DAAL_CHECK_EX(dependentVariableTable->getNumberOfColumns() == 1,
        ErrorIncorrectNumberOfColumns, ArgumentName, dependentVariableStr());
    const Parameter* parameter = static_cast<const Parameter*>(par);
    const size_t nSamplesPerTree(parameter->observationsPerTreeFraction*dataTable->getNumberOfRows());
    DAAL_CHECK_EX(nSamplesPerTree > 0,
        ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());
    const auto nFeatures = dataTable->getNumberOfColumns();
    DAAL_CHECK_EX(parameter->featuresPerNode <= nFeatures,
        ErrorIncorrectParameter, ParameterName, featuresPerNodeStr());
    return s;
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
