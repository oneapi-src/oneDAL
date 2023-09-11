/* file: kmeans_init_partialresult_types.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "algorithms/kmeans/kmeans_init_types.h"
#include "services/daal_defines.h"
#include "src/algorithms/kmeans/kmeans_init_impl.h"
#include "data_management/data/memory_block.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep2LocalPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP2LOCAL_PP_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep3MasterPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP3MASTER_PP_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep4LocalPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP4LOCAL_PP_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedStep5MasterPlusPlusPartialResult, SERIALIZATION_KMEANS_INIT_STEP5MASTER_PP_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Returns a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] id   Identifier of the partial result
 * \return         Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the result table of the K-Means algorithm
* \return Number of features in the result table of the K-Means algorithm
*/
size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr clusters = get(partialClusters);
    return clusters->getNumberOfColumns();
}

#define isPlusPlusMethod(method)                                                                                                     \
    ((method == kmeans::init::plusPlusDense) || (method == kmeans::init::plusPlusCSR) || (method == kmeans::init::parallelPlusDense) \
     || (method == kmeans::init::parallelPlusCSR))

/**
* Checks a partial result of computing initial clusters for the K-Means algorithm
* \param[in] input   %Input object for the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    size_t inputFeatures    = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    const Parameter * kmPar = static_cast<const Parameter *>(par);
    const size_t nClusters  = (isPlusPlusMethod(method) ? 1 : kmPar->nClusters);

    int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    NumericTablePtr pPartialClusters = get(partialClusters);
    if (pPartialClusters.get())
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(pPartialClusters.get(), partialClustersStr(), unexpectedLayouts, 0, inputFeatures, nClusters));
    }
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialClustersNumber).get(), partialClustersNumberStr(), unexpectedLayouts, 0, 1, 1));

    if (dynamic_cast<const Input *>(input))
    {
        DAAL_CHECK_EX(kmPar->nRowsTotal > 0, ErrorIncorrectParameter, ParameterName, nRowsTotalStr());
        DAAL_CHECK_EX(kmPar->nRowsTotal != kmPar->offset, ErrorIncorrectParameter, ParameterName, offsetStr());
    }
    return s;
}

/**
 * Checks a partial result of computing initial clusters for the K-Means algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method of the algorithm
 */
services::Status PartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    const Parameter * kmPar = static_cast<const Parameter *>(par);
    int unexpectedLayouts   = (int)packed_mask;
    const size_t nClusters  = (isPlusPlusMethod(method) ? 1 : kmPar->nClusters);

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialClustersNumber).get(), partialClustersNumberStr(), unexpectedLayouts, 0, 1, 1));
    NumericTablePtr pPartialClusters = get(partialClusters);
    if (pPartialClusters.get())
    {
        const size_t nRows = pPartialClusters->getNumberOfRows();
        DAAL_CHECK_EX(nRows <= nClusters, ErrorIncorrectNumberOfRows, ArgumentName, partialClustersStr());
        s = checkNumericTable(pPartialClusters.get(), partialClustersStr(), unexpectedLayouts);
    }
    return s;
}
DistributedStep2LocalPlusPlusPartialResult::DistributedStep2LocalPlusPlusPartialResult()
    : daal::algorithms::PartialResult(lastDistributedStep2LocalPlusPlusPartialResultDataId + 1)
{}

data_management::NumericTablePtr DistributedStep2LocalPlusPlusPartialResult::get(DistributedStep2LocalPlusPlusPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep2LocalPlusPlusPartialResult::set(DistributedStep2LocalPlusPlusPartialResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

data_management::DataCollectionPtr DistributedStep2LocalPlusPlusPartialResult::get(DistributedStep2LocalPlusPlusPartialResultDataId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedStep2LocalPlusPlusPartialResult::set(DistributedStep2LocalPlusPlusPartialResultDataId id,
                                                     const data_management::DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedStep2LocalPlusPlusPartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                   int method) const
{
    const DistributedStep2LocalPlusPlusParameter * kmPar = static_cast<const DistributedStep2LocalPlusPlusParameter *>(par);
    services::Status s = checkNumericTable(get(outputOfStep2ForStep3).get(), outputOfStep2ForStep3Str(), (int)packed_mask, 0, 1, 1);

    if (kmPar->firstIteration)
        s |= internal::checkLocalData(get(internalResult).get(), kmPar, internalResultStr(), static_cast<const Input *>(input)->get(data).get(),
                                      isParallelPlusMethod(method));
    return s;
}

services::Status DistributedStep2LocalPlusPlusPartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    return services::Status();
}

void DistributedStep2LocalPlusPlusPartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                            int method)
{
    const DistributedStep2LocalPlusPlusParameter * kmPar = static_cast<const DistributedStep2LocalPlusPlusParameter *>(par);
    if (kmPar->firstIteration)
    {
        const auto pLocalData = get(internalResult);
        const auto pNClusters = NumericTable::cast(pLocalData->get(internal::numberOfClusters));
        BlockDescriptor<int> block;
        pNClusters->getBlockOfRows(0, 1, writeOnly, block);
        *block.getBlockPtr() = 0;
        pNClusters->releaseBlockOfRows(block);
    }
}

DistributedStep3MasterPlusPlusPartialResult::DistributedStep3MasterPlusPlusPartialResult()
    : daal::algorithms::PartialResult(lastDistributedStep3MasterPlusPlusPartialResultDataId + 1)
{
    set(outputOfStep3ForStep4, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
    set(rngState, SerializationIfacePtr(new MemoryBlock()));
}

data_management::KeyValueDataCollectionPtr DistributedStep3MasterPlusPlusPartialResult::get(DistributedStep3MasterPlusPlusPartialResultId id) const
{
    return data_management::KeyValueDataCollection::cast(Argument::get(id));
}

data_management::NumericTablePtr DistributedStep3MasterPlusPlusPartialResult::get(DistributedStep3MasterPlusPlusPartialResultId id, size_t key) const
{
    data_management::KeyValueDataCollectionPtr pColl = data_management::KeyValueDataCollection::cast(Argument::get(outputOfStep3ForStep4));
    return data_management::NumericTable::cast((*pColl)[key]);
}

data_management::SerializationIfacePtr DistributedStep3MasterPlusPlusPartialResult::get(DistributedStep3MasterPlusPlusPartialResultDataId id) const
{
    return Argument::get(id);
}

void DistributedStep3MasterPlusPlusPartialResult::add(DistributedStep3MasterPlusPlusPartialResultId id, size_t key,
                                                      const data_management::NumericTablePtr & ptr)
{
    data_management::KeyValueDataCollectionPtr pColl = data_management::KeyValueDataCollection::cast(Argument::get(outputOfStep3ForStep4));
    if (!pColl) return;
    (*pColl)[key] = ptr;
}

services::Status DistributedStep3MasterPlusPlusPartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    int method) const
{
    return this->check(par, method);
}

services::Status DistributedStep3MasterPlusPlusPartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    auto pArg = Argument::get(outputOfStep3ForStep4);
    DAAL_CHECK_EX(pArg.get(), ErrorNullPartialResult, ArgumentName, outputOfStep3ForStep4Str());
    data_management::KeyValueDataCollectionPtr pColl = data_management::KeyValueDataCollection::cast(pArg);
    DAAL_CHECK_EX(pColl.get(), ErrorNullInputDataCollection, ArgumentName, outputOfStep3ForStep4Str());
    pArg = Argument::get(rngState);
    DAAL_CHECK_EX(pArg.get(), ErrorNullPartialResult, ArgumentName, rngStateStr());
    data_management::MemoryBlockPtr pMemBlock = data_management::MemoryBlock::cast(pArg);
    DAAL_CHECK_EX(pMemBlock.get(), ErrorIncorrectItemInDataCollection, ArgumentName, rngStateStr());
    return services::Status();
}

void DistributedStep3MasterPlusPlusPartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                             int method)
{
    auto pColl = get(outputOfStep3ForStep4);
    pColl->clear();
}

DistributedStep4LocalPlusPlusPartialResult::DistributedStep4LocalPlusPlusPartialResult()
    : daal::algorithms::PartialResult(lastDistributedStep4LocalPlusPlusPartialResultId + 1)
{}

data_management::NumericTablePtr DistributedStep4LocalPlusPlusPartialResult::get(DistributedStep4LocalPlusPlusPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep4LocalPlusPlusPartialResult::set(DistributedStep4LocalPlusPlusPartialResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedStep4LocalPlusPlusPartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                   int method) const
{
    const DistributedStep4LocalPlusPlusInput * kmInput = static_cast<const DistributedStep4LocalPlusPlusInput *>(input);
    const auto nFeatures                               = kmInput->get(data)->getNumberOfColumns();
    data_management::NumericTablePtr pInput            = kmInput->get(inputOfStep4FromStep3);
    const auto nRows                                   = pInput->getNumberOfColumns();
    return checkNumericTable(get(outputOfStep4).get(), outputOfStep4Str(), (int)packed_mask, 0, nFeatures, nRows);
}

services::Status DistributedStep4LocalPlusPlusPartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    return services::Status();
}

DistributedStep5MasterPlusPlusPartialResult::DistributedStep5MasterPlusPlusPartialResult()
    : daal::algorithms::PartialResult(lastDistributedStep5MasterPlusPlusPartialResultId + 1)
{}

data_management::NumericTablePtr DistributedStep5MasterPlusPlusPartialResult::get(DistributedStep5MasterPlusPlusPartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep5MasterPlusPlusPartialResult::set(DistributedStep5MasterPlusPlusPartialResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedStep5MasterPlusPlusPartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    int method) const
{
    const Parameter * stepPar                           = static_cast<const Parameter *>(par);
    const DistributedStep5MasterPlusPlusInput * kmInput = static_cast<const DistributedStep5MasterPlusPlusInput *>(input);

    const size_t nMaxCandidates                 = size_t(stepPar->oversamplingFactor * stepPar->nClusters) * stepPar->nRounds + 1;
    const auto pColl                            = kmInput->get(inputCentroids);
    data_management::NumericTablePtr pCentroids = data_management::NumericTable::cast((*pColl)[0]);
    const auto nFeatures                        = pCentroids->getNumberOfColumns();
    services::Status s = checkNumericTable(get(candidates).get(), candidatesStr(), (int)packed_mask, 0, nFeatures, nMaxCandidates);
    s |= checkNumericTable(get(weights).get(), candidateRatingStr(), (int)packed_mask, 0, nMaxCandidates, 1);
    return s;
}

services::Status DistributedStep5MasterPlusPlusPartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    const Parameter * stepPar   = static_cast<const Parameter *>(par);
    const size_t nMaxCandidates = size_t(stepPar->oversamplingFactor * stepPar->nClusters) * stepPar->nRounds + 1;
    services::Status s          = checkNumericTable(get(candidates).get(), candidatesStr(), (int)packed_mask, 0, 0, nMaxCandidates);
    s |= checkNumericTable(get(weights).get(), candidateRatingStr(), (int)packed_mask, 0, nMaxCandidates, 1);
    return s;
}

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
