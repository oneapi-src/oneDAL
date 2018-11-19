/* file: covariance_result.h */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#ifndef __COVARIANCE_RESULT_
#define __COVARIANCE_RESULT_

#include "covariance_types.h"

using namespace daal::data_management;
namespace daal
{
namespace algorithms
{
namespace covariance
{

/**
 * Allocates memory to store final results of the correlation or variance-covariance matrix algorithm
 * \param[in] input     %Input objects of the algorithm
 * \param[in] parameter Parameters of the algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *algInput = static_cast<const Input *>(input);
    size_t nColumns = algInput->getNumberOfFeatures();
    services::Status status;

    set(covariance, HomogenNumericTable<algorithmFPType>::create(nColumns, nColumns, NumericTable::doAllocate, &status));
    set(mean, HomogenNumericTable<algorithmFPType>::create(nColumns, 1, NumericTable::doAllocate, &status));

    return status;
}

/**
 * Allocates memory for storing Covariance final results
 * \param[in] partialResult      Partial Results arguments of the covariance algorithm
 * \param[in] parameter          Parameters of the covariance algorithm
 * \param[in] method             Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method)
{
    const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
    size_t nColumns = pres->getNumberOfFeatures();
    services::Status status;

    set(covariance, HomogenNumericTable<algorithmFPType>::create(nColumns, nColumns, NumericTable::doAllocate, &status));
    set(mean, HomogenNumericTable<algorithmFPType>::create(nColumns, 1, NumericTable::doAllocate, &status));

    return status;
}

} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
