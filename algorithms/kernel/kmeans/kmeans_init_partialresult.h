/* file: kmeans_init_partialresult.h */
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
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_INIT_PARTIALRESULT_
#define __KMEANS_INIT_PARTIALRESULT_

#include "algorithms/kmeans/kmeans_init_types.h"
#include "kmeans_init_impl.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{

#define isPlusPlusMethod(method)\
    ((method == kmeans::init::plusPlusDense) || (method == kmeans::init::plusPlusCSR) || \
    (method == kmeans::init::parallelPlusDense) || (method == kmeans::init::parallelPlusCSR))

/**
 * Allocates memory to store partial results of computing initial clusters for the K-Means algorithm
 * \param[in] input     Pointer to the input structure
 * \param[in] parameter Pointer to the parameter structure
 * \param[in] method    Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status status;
    set(partialClustersNumber, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    return status;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedStep2LocalPlusPlusPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status status;
    const DistributedStep2LocalPlusPlusParameter *kmPar = static_cast<const DistributedStep2LocalPlusPlusParameter *>(parameter);

    set(outputOfStep2ForStep3, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));

    if(isParallelPlusMethod(method) && kmPar->outputForStep5Required)
    {
        const size_t nMaxCandidates = size_t(kmPar->oversamplingFactor*kmPar->nClusters)*kmPar->nRounds + 1;
        set(outputOfStep2ForStep5, HomogenNumericTable<int>::create(nMaxCandidates, 1, NumericTable::doAllocate, &status));
    }
    if(!kmPar->firstIteration)
        return services::Status();

    DataCollectionPtr pLocalData(new DataCollection(
        isParallelPlusMethod(method) ? internal::localDataSize : internal::localDataSize - 1));
    set(internalResult, pLocalData);
    auto pData = static_cast<const Input *>(input)->get(data);
    const auto nRows = pData->getNumberOfRows();
    (*pLocalData)[internal::closestClusterDistance] = HomogenNumericTable<algorithmFPType>::create(nRows, 1, NumericTable::doAllocate, &status);
    (*pLocalData)[internal::closestCluster] = HomogenNumericTable<int>::create(nRows, 1, NumericTable::doAllocate, &status);
    (*pLocalData)[internal::numberOfClusters] = HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status);
    if(isParallelPlusMethod(method))
    {
        const size_t nMaxCandidates = size_t(kmPar->oversamplingFactor*kmPar->nClusters)*kmPar->nRounds + 1;
        (*pLocalData)[internal::candidateRating] = HomogenNumericTable<int>::create(nMaxCandidates, 1, NumericTable::doAllocate, &status);
    }
    return status;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedStep3MasterPlusPlusPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    //nothing to allocate
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedStep4LocalPlusPlusPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status status;
    const DistributedStep4LocalPlusPlusInput* kmInput = static_cast<const DistributedStep4LocalPlusPlusInput*>(input);
    const auto nFeatures = kmInput->get(data)->getNumberOfColumns();
    NumericTablePtr pInput = kmInput->get(inputOfStep4FromStep3);
    NumericTablePtr pOutput = HomogenNumericTable<algorithmFPType>::create(DictionaryIface::FeaturesEqual::equal, nFeatures, pInput->getNumberOfColumns(), NumericTable::doAllocate, &status);
    set(outputOfStep4, pOutput);
    return status;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedStep5MasterPlusPlusPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status status;
    const Parameter *stepPar = static_cast<const Parameter *>(parameter);
    const DistributedStep5MasterPlusPlusInput* kmInput = static_cast<const DistributedStep5MasterPlusPlusInput*>(input);

    const size_t nMaxCandidates = size_t(stepPar->oversamplingFactor*stepPar->nClusters)*stepPar->nRounds + 1;
    const auto pColl = kmInput->get(inputCentroids);
    NumericTablePtr pCentroids = NumericTable::cast((*pColl)[0]);
    const auto nFeatures = pCentroids->getNumberOfColumns();
    NumericTablePtr pCandidates = HomogenNumericTable<algorithmFPType>::create(nFeatures, nMaxCandidates, NumericTable::doAllocate, &status);
    set(candidates, pCandidates);
    NumericTablePtr pCandidateWeights = HomogenNumericTable<algorithmFPType>::create(nMaxCandidates, 1, NumericTable::doAllocate, &status);
    set(weights, pCandidateWeights);
    return status;
}

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
