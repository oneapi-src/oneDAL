/* file: binary_confusion_matrix.h */
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

#ifndef __BINARY_CONFUSION_MATRIX_H__
#define __BINARY_CONFUSION_MATRIX_H__

#include "binary_confusion_matrix_types.h"

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


/**
 * Allocates memory for storing results of the quality metric algorithm
 * \param[in] input     Pointer to the input objects structure
 * \param[in] parameter Pointer to the parameter structure
 * \param[in] method    Computation method of the algorithm
 */
template<typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    set(confusionMatrix, data_management::NumericTablePtr(
            new data_management::HomogenNumericTable<algorithmFPType>(2, 2, data_management::NumericTableIface::doAllocate)));
    set(binaryMetrics, data_management::NumericTablePtr(
            new data_management::HomogenNumericTable<algorithmFPType>(6, 1, data_management::NumericTableIface::doAllocate)));
}

} // namespace binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
