/* file: binary_confusion_matrix_types.cpp */
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
//  Declaration of data types for computing the binary confusion matrix.
//--
*/


#include "binary_confusion_matrix_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CLASSIFIER_BINARY_CONFUSION_MATRIX_RESULT_ID);
Parameter::Parameter(double beta) : beta(beta) {}

Status Parameter::check() const
{
    DAAL_CHECK_EX(beta > 0, ErrorIncorrectParameter, ParameterName, betaStr());
    return Status();
}


Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

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
Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    NumericTablePtr predictedLabelsTable = get(predictedLabels);
    NumericTablePtr groundTruthLabelsTable = get(groundTruthLabels);

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(predictedLabelsTable.get(), predictedLabelsStr(), unexpectedLayouts, 0, 1));

    const size_t nRows = predictedLabelsTable->getNumberOfRows();
    return checkNumericTable(groundTruthLabelsTable.get(), groundTruthLabelsStr(), unexpectedLayouts, 0, 1, nRows);
}


Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

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
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    NumericTablePtr confusionMatrixTable = get(confusionMatrix);
    NumericTablePtr binaryMetricsTable = get(binaryMetrics);
    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(confusionMatrixTable.get(), confusionMatrixStr(), unexpectedLayouts, 0, 2, 2));
    return checkNumericTable(binaryMetricsTable.get(), binaryMetricsStr(), unexpectedLayouts, 0, 6, 1);
}

} // namespace interface1
} // binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal
