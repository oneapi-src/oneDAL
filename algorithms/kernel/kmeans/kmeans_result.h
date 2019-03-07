/* file: kmeans_result.h */
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

#ifndef __KMEANS_RESULT_
#define __KMEANS_RESULT_

#include "algorithms/kmeans/kmeans_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{

/**
 * Allocates memory to store the results of the K-Means algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *kmPar = static_cast<const Parameter *>(parameter);

    Input *algInput  = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->getNumberOfFeatures();
    size_t nClusters = kmPar->nClusters;

    services::Status status;
    set(centroids, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    set(objectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    set(nIterations, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    if(kmPar->assignFlag)
    {
        size_t nRows = algInput->get(data)->getNumberOfRows();
        set(assignments, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
    }

    return status;
}
/**
 * Allocates memory to store the results of the K-Means algorithm
 * \param[in] partialResult Pointer to the partial result structure
 * \param[in] parameter     Pointer to the structure of the algorithm parameters
 * \param[in] method        Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method)
{
    size_t nClusters = (static_cast<const Parameter *>(parameter))->nClusters;
    size_t nFeatures = (static_cast<const PartialResult *>(partialResult))->getNumberOfFeatures();

    services::Status status;
    set(centroids, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    set(objectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    set(nIterations, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    return status;
}

} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
