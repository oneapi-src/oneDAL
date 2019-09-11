/* file: ridge_regression_training_input.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of ridge regression algorithm classes.
//--
*/

#include "algorithms/ridge_regression/ridge_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace interface1
{

Input::Input() : linear_model::training::Input(lastInputId + 1) {}
Input::Input(const Input& other) : linear_model::training::Input(other){}

/**
 * Returns an input object for ridge regression model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return linear_model::training::Input::get(linear_model::training::InputId(id));
}

/**
 * Sets an input object for ridge regression model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    linear_model::training::Input::set(linear_model::training::InputId(id), value);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNumberOfFeatures() const { return get(data)->getNumberOfColumns(); }

/**
* Returns the number of dependent variables
* \return Number of dependent variables
*/
size_t Input::getNumberOfDependentVariables() const { return get(dependentVariables)->getNumberOfColumns(); }

/**
* Checks an input object for the ridge regression algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*
 * \return Status of computations
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Input::check(par, method));

    const NumericTablePtr dataTable = get(data);
    size_t nRowsInData = dataTable->getNumberOfRows();
    size_t nColumnsInData = dataTable->getNumberOfColumns();

    DAAL_CHECK(nRowsInData >= nColumnsInData, ErrorIncorrectNumberOfObservations);

    const NumericTablePtr dependentVariableTable = get(dependentVariables);
    const size_t nColumnsInDepVariable = dependentVariableTable->getNumberOfColumns();

    TrainParameter *trainParameter   = static_cast<TrainParameter *>(const_cast<daal::algorithms::Parameter *>(par));
    DAAL_CHECK_STATUS(s, trainParameter->check());

    size_t ridgeParamsNumberOfColumns = trainParameter->ridgeParameters->getNumberOfColumns();
    DAAL_CHECK((ridgeParamsNumberOfColumns == 1) || (nColumnsInDepVariable == ridgeParamsNumberOfColumns), ErrorIncorrectNumberOfColumns);
    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
