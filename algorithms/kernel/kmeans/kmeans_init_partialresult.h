/* file: kmeans_init_partialresult.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    Argument::set(partialClustersNumber, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<int>( 1, 1, data_management::NumericTable::doAllocate)));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedStep2LocalPlusPlusPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedStep2LocalPlusPlusParameter *kmPar = static_cast<const DistributedStep2LocalPlusPlusParameter *>(parameter);

    Argument::set(outputOfStep2ForStep3, data_management::SerializationIfacePtr(
        new data_management::HomogenNumericTable<algorithmFPType>(1, 1, data_management::NumericTable::doAllocate)));

    if(isParallelPlusMethod(method) && kmPar->outputForStep5Required)
    {
        const size_t nMaxCandidates = size_t(kmPar->oversamplingFactor*kmPar->nClusters)*kmPar->nRounds + 1;
        Argument::set(outputOfStep2ForStep5, data_management::SerializationIfacePtr(
            new data_management::HomogenNumericTable<int>(nMaxCandidates, 1, data_management::NumericTable::doAllocate)));
    }
    if(!kmPar->firstIteration)
        return services::Status();

    data_management::DataCollectionPtr pLocalData(new data_management::DataCollection(
        isParallelPlusMethod(method) ? internal::localDataSize : internal::localDataSize - 1));
    set(internalResult, pLocalData);
    auto pData = static_cast<const Input *>(input)->get(data);
    const auto nRows = pData->getNumberOfRows();
    (*pLocalData)[internal::closestClusterDistance].reset(new data_management::HomogenNumericTable<algorithmFPType>(nRows, 1, data_management::NumericTable::doAllocate));
    (*pLocalData)[internal::closestCluster].reset(new data_management::HomogenNumericTable<int>(nRows, 1, data_management::NumericTable::doAllocate));
    (*pLocalData)[internal::numberOfClusters].reset(new data_management::HomogenNumericTable<int>(1, 1, data_management::NumericTable::doAllocate));
    if(isParallelPlusMethod(method))
    {
        const size_t nMaxCandidates = size_t(kmPar->oversamplingFactor*kmPar->nClusters)*kmPar->nRounds + 1;
        (*pLocalData)[internal::candidateRating].reset(new data_management::HomogenNumericTable<int>(nMaxCandidates, 1, data_management::NumericTable::doAllocate));
    }
    return services::Status();
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
    const DistributedStep4LocalPlusPlusInput* kmInput = static_cast<const DistributedStep4LocalPlusPlusInput*>(input);
    const auto nFeatures = kmInput->get(data)->getNumberOfColumns();
    data_management::NumericTablePtr pInput = kmInput->get(inputOfStep4FromStep3);
    data_management::NumericTablePtr pOutput(new data_management::HomogenNumericTable<algorithmFPType>(data_management::DictionaryIface::FeaturesEqual::equal, nFeatures,
        pInput->getNumberOfColumns(), data_management::NumericTable::doAllocate));
    set(outputOfStep4, pOutput);
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedStep5MasterPlusPlusPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *stepPar = static_cast<const Parameter *>(parameter);
    const DistributedStep5MasterPlusPlusInput* kmInput = static_cast<const DistributedStep5MasterPlusPlusInput*>(input);

    const size_t nMaxCandidates = size_t(stepPar->oversamplingFactor*stepPar->nClusters)*stepPar->nRounds + 1;
    const auto pColl = kmInput->get(inputCentroids);
    data_management::NumericTablePtr pCentroids = data_management::NumericTable::cast((*pColl)[0]);
    const auto nFeatures = pCentroids->getNumberOfColumns();
    data_management::NumericTablePtr pCandidates(new data_management::HomogenNumericTable<algorithmFPType>(nFeatures,
        nMaxCandidates, data_management::NumericTable::doAllocate));
    set(candidates, pCandidates);
    data_management::NumericTablePtr pCandidateWeights(new data_management::HomogenNumericTable<algorithmFPType>(nMaxCandidates, 1,
        data_management::NumericTable::doAllocate));
    set(weights, pCandidateWeights);
    return services::Status();
}

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
