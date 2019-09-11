/* file: kmeans_partialresult.h */
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
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_PARTIALRESULT_
#define __KMEANS_PARTIALRESULT_

#include "algorithms/kmeans/kmeans_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{

/**
 * Allocates memory to store partial results of the K-Means algorithm
 * \param[in] input        Pointer to the structure of the input objects
 * \param[in] parameter    Pointer to the structure of the algorithm parameters
 * \param[in] method       Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *kmPar = static_cast<const Parameter *>(parameter);

    size_t nFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    size_t nClusters = kmPar->nClusters;

    services::Status status;
    set(nObservations, HomogenNumericTable<algorithmFPType>::create(1, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialSums, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialObjectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialCandidatesDistances, HomogenNumericTable<algorithmFPType>::create(1, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialCandidatesCentroids, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);

    const Input *step1Input = dynamic_cast<const Input *>(input);
    if (kmPar->assignFlag && step1Input)
    {
        const size_t nRows = step1Input->get(data)->getNumberOfRows();
        set(partialAssignments, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
    }

    return status;
}

} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
