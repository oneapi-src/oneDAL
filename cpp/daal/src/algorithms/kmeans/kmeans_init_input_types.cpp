/* file: kmeans_init_input_types.cpp */
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
#include "data_management/data/row_merged_numeric_table.h"
#include "data_management/data/memory_block.h"
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
Input::Input() : InputIface(lastInputId + 1) {}
Input::Input(size_t nElements) : InputIface(nElements) {};

/**
* Returns input objects for computing initial clusters for the K-Means algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for computing initial clusters for the K-Means algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the input object
*/
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the Input data table
* \return Number of features in the Input data table
*/
size_t Input::getNumberOfFeatures() const
{
    NumericTablePtr inTable = get(data);
    return inTable->getNumberOfColumns();
}

static bool isCSRMethod(int method)
{
    return (method == kmeans::init::deterministicCSR || method == kmeans::init::randomCSR || method == kmeans::init::plusPlusCSR
            || method == kmeans::init::parallelPlusCSR);
}

/**
* Checks an input object for computing initial clusters for the K-Means algorithm
* \param[in] par     %Input object
* \param[in] method  Method of the algorithm
*/
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    if (isParallelPlusMethod(method))
    {
        //check parallel plus method parameters
        const Parameter * prm = (const Parameter *)(parameter);
        DAAL_CHECK_EX(prm->oversamplingFactor > 0, ErrorIncorrectParameter, ParameterName, oversamplingFactorStr());

        DAAL_CHECK_EX(prm->nRounds > 0, ErrorIncorrectParameter, ParameterName, nRoundsStr());
        size_t L(prm->oversamplingFactor * prm->nClusters);
        if (L * prm->nRounds <= prm->nClusters)
        {
            return services::Status(services::Error::create(ErrorIncorrectParameter, ParameterName, nRoundsStr()));
        }
    }
    services::Status s;
    if (isCSRMethod(method))
    {
        const int expectedLayout = (int)NumericTableIface::csrArray;
        s                        = checkNumericTable(get(data).get(), dataStr(), 0, expectedLayout);
    }
    else
    {
        s = checkNumericTable(get(data).get(), dataStr());
    }
    return s;
}

DistributedStep2LocalPlusPlusInput::DistributedStep2LocalPlusPlusInput() : Input(lastDistributedStep2LocalPlusPlusInputId + 1) {}
DistributedStep2LocalPlusPlusInput::DistributedStep2LocalPlusPlusInput(const DistributedStep2LocalPlusPlusInput & o)
{
    for (size_t i = 0; i < size(); ++i) Argument::set(i, o.Argument::get(i));
}

NumericTablePtr DistributedStep2LocalPlusPlusInput::get(InputId id) const
{
    return Input::get(id);
}

void DistributedStep2LocalPlusPlusInput::set(InputId id, const NumericTablePtr & ptr)
{
    Input::set(id, ptr);
}

DataCollectionPtr DistributedStep2LocalPlusPlusInput::get(DistributedLocalPlusPlusInputDataId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedStep2LocalPlusPlusInput::set(DistributedLocalPlusPlusInputDataId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedStep2LocalPlusPlusInput::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = Input::check(par, method);
    if (!s) return s;

    const auto nFeatures                                   = get(data)->getNumberOfColumns();
    const DistributedStep2LocalPlusPlusParameter * stepPar = (const DistributedStep2LocalPlusPlusParameter *)(par);
    const size_t nRows = (stepPar->firstIteration || !(isParallelPlusMethod(method)) ? 1 : size_t(stepPar->oversamplingFactor * stepPar->nClusters));
    s                  = checkNumericTable(get(inputOfStep2).get(), inputOfStep2Str(), (int)packed_mask, 0, nFeatures, nRows);
    if (!stepPar->firstIteration)
        s = internal::checkLocalData(get(internalInput).get(), stepPar, internalInputStr(), get(data).get(), isParallelPlusMethod(method));
    return s;
}

NumericTablePtr DistributedStep2LocalPlusPlusInput::get(DistributedStep2LocalPlusPlusInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep2LocalPlusPlusInput::set(DistributedStep2LocalPlusPlusInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

DistributedStep3MasterPlusPlusInput::DistributedStep3MasterPlusPlusInput() : Input(lastDistributedStep3MasterPlusPlusInputId + 1)
{
    set(inputOfStep3FromStep2, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

DistributedStep3MasterPlusPlusInput::DistributedStep3MasterPlusPlusInput(const DistributedStep3MasterPlusPlusInput & o)
{
    set(inputOfStep3FromStep2, o.get(inputOfStep3FromStep2));
}

KeyValueDataCollectionPtr DistributedStep3MasterPlusPlusInput::get(DistributedStep3MasterPlusPlusInputId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

void DistributedStep3MasterPlusPlusInput::set(DistributedStep3MasterPlusPlusInputId id, const KeyValueDataCollectionPtr & ptr)
{
    Input::set(id, ptr);
}

void DistributedStep3MasterPlusPlusInput::add(DistributedStep3MasterPlusPlusInputId id, size_t key, const NumericTablePtr & ptr)
{
    KeyValueDataCollectionPtr pColl = get(inputOfStep3FromStep2);
    (*pColl)[key]                   = ptr;
}

services::Status DistributedStep3MasterPlusPlusInput::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = Input::check(par, method);
    if (!s) return s;
    KeyValueDataCollectionPtr pColl = get(inputOfStep3FromStep2);
    DAAL_CHECK_EX(pColl.get(), ErrorNullInputDataCollection, ArgumentName, inputOfStep3FromStep2Str());
    DAAL_CHECK_EX(pColl->size() > 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, inputOfStep3FromStep2Str());
    for (size_t i = 0; i < pColl->size(); ++i)
    {
        SerializationIfacePtr pVal = pColl->getValueByIndex(i);
        DAAL_CHECK_EX(pVal.get(), ErrorNullInput, ArgumentName, inputOfStep3FromStep2Str());
        NumericTablePtr pTbl = NumericTable::cast(pVal);
        s |= checkNumericTable(pTbl.get(), inputOfStep3FromStep2Str(), (int)packed_mask, 0, 1, 1);
    }
    return s;
}

DistributedStep4LocalPlusPlusInput::DistributedStep4LocalPlusPlusInput() : Input(lastDistributedStep4LocalPlusPlusInputId + 1) {}
DistributedStep4LocalPlusPlusInput::DistributedStep4LocalPlusPlusInput(const DistributedStep4LocalPlusPlusInput & o)
{
    for (size_t i = 0; i < size(); ++i) Argument::set(i, o.Argument::get(i));
}

NumericTablePtr DistributedStep4LocalPlusPlusInput::get(InputId id) const
{
    return Input::get(id);
}

void DistributedStep4LocalPlusPlusInput::set(InputId id, const NumericTablePtr & ptr)
{
    Input::set(id, ptr);
}

DataCollectionPtr DistributedStep4LocalPlusPlusInput::get(DistributedLocalPlusPlusInputDataId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedStep4LocalPlusPlusInput::set(DistributedLocalPlusPlusInputDataId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

NumericTablePtr DistributedStep4LocalPlusPlusInput::get(DistributedStep4LocalPlusPlusInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedStep4LocalPlusPlusInput::set(DistributedStep4LocalPlusPlusInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedStep4LocalPlusPlusInput::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = Input::check(par, method);
    if (!s) return s;
    const Parameter * stepPar = (const Parameter *)(par);
    s = internal::checkLocalData(get(internalInput).get(), stepPar, internalInputStr(), get(data).get(), isParallelPlusMethod(method));
    s |= checkNumericTable(get(inputOfStep4FromStep3).get(), inputOfStep4FromStep3Str(), (int)packed_mask, 0, 0, 1);
    return s;
}

DistributedStep5MasterPlusPlusInput::DistributedStep5MasterPlusPlusInput() : Input(lastDistributedStep5MasterPlusPlusInputDataId + 1)
{
    set(inputCentroids, DataCollectionPtr(new DataCollection()));
    set(inputOfStep5FromStep2, DataCollectionPtr(new DataCollection()));
}

DistributedStep5MasterPlusPlusInput::DistributedStep5MasterPlusPlusInput(const DistributedStep5MasterPlusPlusInput & o)
{
    for (size_t i = 0; i < size(); ++i) Argument::set(i, o.Argument::get(i));
}

DataCollectionPtr DistributedStep5MasterPlusPlusInput::get(DistributedStep5MasterPlusPlusInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedStep5MasterPlusPlusInput::set(DistributedStep5MasterPlusPlusInputId id, const DataCollectionPtr & ptr)
{
    daal::algorithms::Input::set(id, ptr);
}

SerializationIfacePtr DistributedStep5MasterPlusPlusInput::get(DistributedStep5MasterPlusPlusInputDataId id) const
{
    return Argument::get(id);
}

void DistributedStep5MasterPlusPlusInput::set(DistributedStep5MasterPlusPlusInputDataId id, const SerializationIfacePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedStep5MasterPlusPlusInput::add(DistributedStep5MasterPlusPlusInputId id, const NumericTablePtr & ptr)
{
    DataCollectionPtr pColl = get(id);
    if (pColl.get()) pColl->push_back(ptr);
}

services::Status DistributedStep5MasterPlusPlusInput::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = Input::check(par, method);
    if (!s) return s;
    const Parameter * stepPar   = (const Parameter *)(par);
    const size_t nMaxCandidates = size_t(stepPar->oversamplingFactor * stepPar->nClusters) * stepPar->nRounds + 1;
    for (size_t i = 0; i < 2; ++i)
    {
        size_t nCandidates   = 0;
        const auto pArg      = Argument::get(i ? inputOfStep5FromStep2 : inputCentroids);
        const char * argName = (i ? inputOfStep5FromStep2Str() : inputCentroidsStr());
        DAAL_CHECK_EX(pArg.get(), ErrorNullInput, ArgumentName, argName);
        DataCollectionPtr pColl = DataCollection::cast(pArg);
        DAAL_CHECK_EX(pColl.get(), ErrorNullInputDataCollection, ArgumentName, argName);
        DAAL_CHECK_EX(pColl->size() > 0, ErrorIncorrectNumberOfElementsInInputCollection, ArgumentName, argName);

        for (size_t j = 0; j < pColl->size(); ++j)
        {
            const auto pItem = (*pColl)[j];
            DAAL_CHECK_EX(pItem.get(), ErrorNullInputNumericTable, ArgumentName, argName);
            const NumericTablePtr pTbl = NumericTable::cast(pItem);
            DAAL_CHECK_EX(pTbl.get(), ErrorIncorrectItemInDataCollection, ArgumentName, argName);

            s |= checkNumericTable(pTbl.get(), argName, (int)packed_mask, 0, i ? nMaxCandidates : 0, i ? 1 : 0);
            if (!s) return s;
            if (i == 0) nCandidates += pTbl->getNumberOfRows();
        }
        if (i == 0) DAAL_CHECK(nCandidates == nMaxCandidates, ErrorIncorrectTotalNumberOfPartialClusters);
    }
    const auto pArg = Argument::get(inputOfStep5FromStep3);
    DAAL_CHECK_EX(pArg.get(), ErrorNullInput, ArgumentName, inputOfStep5FromStep3Str());
    MemoryBlockPtr pMemBlock = MemoryBlock::cast(pArg);
    DAAL_CHECK_EX(pMemBlock.get(), ErrorIncorrectItemInDataCollection, ArgumentName, rngStateStr());
    return s;
}

namespace internal
{
services::Status checkLocalData(const DataCollection * pInput, const Parameter * par, const char * dataName, const NumericTable * pData,
                                bool bParallelPlus)
{
    const auto nRows = pData->getNumberOfRows();
    DAAL_CHECK_EX(pInput != nullptr, ErrorIncorrectInputNumericTable, ArgumentName, dataName); //TODO
    DAAL_CHECK_EX(pInput->size() == (bParallelPlus ? localDataSize : localDataSize - 1), ErrorIncorrectDataCollectionSize, ArgumentName, dataName);
    const auto pClosestClusterDistance = NumericTable::cast(pInput->get(closestClusterDistance));
    services::Status s                 = checkNumericTable(pClosestClusterDistance.get(), closestClusterDistanceStr(), (int)packed_mask, 0, nRows, 1);

    const auto pClosestCluster = NumericTable::cast(pInput->get(closestCluster));
    s |= checkNumericTable(pClosestCluster.get(), closestClusterStr(), (int)packed_mask, 0, nRows, 1);

    const auto pNClusters = NumericTable::cast(pInput->get(numberOfClusters));
    s |= checkNumericTable(pNClusters.get(), numberOfClustersStr(), (int)packed_mask, 0, 1, 1);

    if (!bParallelPlus) return s;

    const size_t nMaxCandidates = size_t(par->oversamplingFactor * par->nClusters) * par->nRounds + 1;
    const auto pRating          = NumericTable::cast(pInput->get(candidateRating));
    s |= checkNumericTable(pRating.get(), candidateRatingStr(), (int)packed_mask, 0, nMaxCandidates, 1);
    return s;
}

} // namespace internal

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
