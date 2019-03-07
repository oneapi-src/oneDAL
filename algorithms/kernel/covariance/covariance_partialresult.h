/* file: covariance_partialresult.h */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#ifndef __COVARIANCE_PARTIALRESULT_
#define __COVARIANCE_PARTIALRESULT_

#include "covariance_types.h"

using namespace daal::data_management;
namespace daal
{
namespace algorithms
{
namespace covariance
{

/**
 * Allocates memory to store partial results of the correlation or variance-covariance matrix algorithm
 * \param[in] input     %Input objects of the algorithm
 * \param[in] parameter Parameters of the algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const InputIface *algInput = static_cast<const InputIface *>(input);
    size_t nColumns = algInput->getNumberOfFeatures();
    services::Status status;

    set(nObservations, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate, &status));
    set(crossProduct, HomogenNumericTable<algorithmFPType>::create(nColumns, nColumns, NumericTable::doAllocate, &status));
    set(sum, HomogenNumericTable<algorithmFPType>::create(nColumns, 1, NumericTable::doAllocate, &status));
    return status;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const InputIface *algInput = static_cast<const InputIface *>(input);
    size_t nColumns = algInput->getNumberOfFeatures();

    get(nObservations)->assign((algorithmFPType)0.0);
    get(crossProduct)->assign((algorithmFPType)0.0);
    get(sum)->assign((algorithmFPType)0.0);
    return services::Status();
}

} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
