/* file: linear_regression_training_input.cpp */
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
//  Implementation of linear regression algorithm classes.
//--
*/

#include "algorithms/linear_regression/linear_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace interface1
{

/** Default constructor */
Input::Input() : linear_model::training::Input(lastInputId + 1) {}
Input::Input(const Input& other) : linear_model::training::Input(other){}

/**
 * Returns an input object for linear regression model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return linear_model::training::Input::get(linear_model::training::InputId(id));
}

/**
 * Sets an input object for linear regression model-based training
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

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Input::check(par, method));

    const Parameter *parameter = static_cast<const Parameter *>(par);

    NumericTablePtr dataTable = get(data);
    size_t nRowsInData = dataTable->getNumberOfRows();
    size_t nColumnsInData = dataTable->getNumberOfColumns();
    DAAL_CHECK(nRowsInData >= nColumnsInData + (int)(method == qrDense && parameter->interceptFlag == true), ErrorIncorrectNumberOfRows);
    return s;
}

} // namespace interface1
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
