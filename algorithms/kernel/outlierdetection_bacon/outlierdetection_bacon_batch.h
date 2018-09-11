/* file: outlierdetection_bacon_batch.h */
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
//  Outlier Detection algorithm parameter structure
//--
*/

#ifndef __OUTLIERDETECTION_BACON_BATCH_H__
#define __OUTLIERDETECTION_BACON_BATCH_H__

#include "outlier_detection_bacon_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace bacon_outlier_detection
{
namespace interface1
{

/**
 * Allocates memory to store the results of the multivariate outlier detection algorithm
 * \tparam algorithmFPType  Data type to use for storing results, double or float
 * \param[in] input   Pointer to %Input objects of the algorithm
 * \param[in] parameter     Pointer to the parameters of the algorithm
 * \param[in] method  Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nVectors = algInput->get(data)->getNumberOfRows();
    set(weights, HomogenNumericTable<algorithmFPType>::create(1, nVectors, NumericTable::doAllocate, &s));
    return s;
}

} // namespace interface1
} // namespace bacon_outlier_detection
} // namespace algorithms
} // namespace daal

#endif
