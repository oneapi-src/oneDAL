/* file: binary_confusion_matrix_types.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Declaration of data types for computing the binary confusion matrix.
//--
*/


#include "binary_confusion_matrix_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace quality_metric
{
namespace binary_confusion_matrix
{
namespace interface1
{

Parameter::Parameter(double beta) : beta(beta) {}

void Parameter::check() const
{
    DAAL_CHECK_EX(beta > 0, ErrorIncorrectParameter, ParameterName, betaStr());
}


Input::Input() : daal::algorithms::Input(2) {}

/**
 * Returns an input object of the quality metric
 * \param[in] id   Identifier of the input object
 * \return         %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object of the quality metric
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of an input object
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    NumericTablePtr predictedLabelsTable = get(predictedLabels);
    NumericTablePtr groundTruthLabelsTable = get(groundTruthLabels);

    int unexpectedLayouts = (int)packed_mask;
    if(!checkNumericTable(predictedLabelsTable.get(), this->_errors.get(), predictedLabelsStr(), unexpectedLayouts, 0, 1)) { return; }

    size_t nRows = predictedLabelsTable->getNumberOfRows();
    if(!checkNumericTable(groundTruthLabelsTable.get(), this->_errors.get(), groundTruthLabelsStr(), unexpectedLayouts, 0, 1, nRows)) { return; }
}


Result::Result() : daal::algorithms::Result(2) {}

/**
 * Returns the quality metric of the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \return          Quality metric of the classification algorithm
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the binary confusion matrix algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \param[in] value Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    NumericTablePtr confusionMatrixTable = get(confusionMatrix);
    NumericTablePtr binaryMetricsTable = get(binaryMetrics);
    int unexpectedLayouts = (int)packed_mask;

    if(!checkNumericTable(confusionMatrixTable.get(), this->_errors.get(), confusionMatrixStr(), unexpectedLayouts, 0, 2, 2)) { return; }
    if(!checkNumericTable(binaryMetricsTable.get(), this->_errors.get(), binaryMetricsStr(), unexpectedLayouts, 0, 6, 1)) { return; }
}

} // namespace interface1
} // binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal
