/* file: moments_batch.h */
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
//  Implementation of LowOrderMoments algorithm and types methods.
//--
*/
#ifndef __MOMENTS_BATCH__
#define __MOMENTS_BATCH__

#include "low_order_moments_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{

/**
 * Allocates memory for storing final results of the low order %moments algorithm
 * \param[in] input     Pointer to the structure with result objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    size_t nFeatures = 0;
    DAAL_CHECK_STATUS(s, static_cast<const InputIface *>(input)->getNumberOfColumns(nFeatures));

    for(size_t i = 0; i < lastResultId + 1; i++)
    {
        Argument::set(i, HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, NumericTable::doAllocate, &s));
    }
    return s;
}

/**
 * Allocates memory for storing final results of the low order %moments algorithm
 * \param[in] partialResult     Pointer to the structure with partial result objects
 * \param[in] parameter         Pointer to the structure of algorithm parameters
 * \param[in] method            Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const int method)
{
    size_t nFeatures;
    services::Status s;
    DAAL_CHECK_STATUS(s, static_cast<const PartialResult *>(partialResult)->getNumberOfColumns(nFeatures));
    for(size_t i = 0; i < lastResultId + 1; i++)
    {
        Argument::set(i, HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, NumericTable::doAllocate, &s));
    }
    return s;
}

}// namespace low_order_moments
}// namespace algorithms
}// namespace daal

#endif
