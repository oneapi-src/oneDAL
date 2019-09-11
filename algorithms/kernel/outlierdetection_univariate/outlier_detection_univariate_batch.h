/* file: outlier_detection_univariate_batch.h */
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
//  Outlier Detection algorithm parameter structure
//--
*/

#ifndef __OUTLIERDETECTION_UNIVARIATE_BATCH_H__
#define __OUTLIERDETECTION_UNIVARIATE_BATCH_H__

#include "outlier_detection_univariate_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace interface1
{

/**
 * Registers user-allocated memory to store univariate outlier detection results
 * \param[in] input   Pointer to the %input objects for the algorithm
 * \param[in] parameter     Pointer to the parameters of the algorithm
 * \param[in] method  univariate outlier detection computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();
    size_t nVectors  = algInput->get(data)->getNumberOfRows();
    set(weights, HomogenNumericTable<algorithmFPType>::create(nFeatures, nVectors, NumericTable::doAllocate, &s));
    return s;
}

} // namespace interface1
} // namespace univariate_outlier_detection
} // namespace algorithms
} // namespace daal

#endif
