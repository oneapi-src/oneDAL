/* file: multiclass_confusion_matrix.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of data types for computing the multiclass confusion matrix.
//--
*/

#ifndef __MULTICLASS_CONFUSION_MATRIX_H__
#define __MULTICLASS_CONFUSION_MATRIX_H__

#include "multiclass_confusion_matrix_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace quality_metric
{
namespace multiclass_confusion_matrix
{
/**
 * Allocates memory for storing the computed quality metric
 * \param[in] input     Pointer to the input structure
 * \param[in] parameter Pointer to the parameter structure
 * \param[in] method    Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status st;
    const Parameter * classifierParam = static_cast<const Parameter *>(parameter);
    size_t nClasses                   = classifierParam->nClasses;
    set(confusionMatrix,
        data_management::HomogenNumericTable<algorithmFPType>::create(nClasses, nClasses, data_management::NumericTableIface::doAllocate, &st));

    set(multiClassMetrics, data_management::HomogenNumericTable<algorithmFPType>::create(8, 1, data_management::NumericTableIface::doAllocate, &st));
    return st;
}

} // namespace multiclass_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
