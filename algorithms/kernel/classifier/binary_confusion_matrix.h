/* file: binary_confusion_matrix.h */
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status st;
    set(confusionMatrix, data_management::HomogenNumericTable<algorithmFPType>::create(2, 2, data_management::NumericTableIface::doAllocate, &st));
    set(binaryMetrics, data_management::HomogenNumericTable<algorithmFPType>::create(6, 1, data_management::NumericTableIface::doAllocate, &st));
    return st;
}

} // namespace binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
