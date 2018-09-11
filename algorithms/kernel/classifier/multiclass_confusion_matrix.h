/* file: multiclass_confusion_matrix.h */
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
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status st;
    const Parameter *classifierParam = static_cast<const Parameter *>(parameter);
    size_t nClasses = classifierParam->nClasses;
    set(confusionMatrix, data_management::HomogenNumericTable<algorithmFPType>::create(nClasses, nClasses, data_management::NumericTableIface::doAllocate, &st));

    set(multiClassMetrics, data_management::HomogenNumericTable<algorithmFPType>::create(8, 1, data_management::NumericTableIface::doAllocate, &st));
    return st;
}

} // namespace multiclass_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
