/* file: implicit_als_train_init_partial_result.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_init_types.h"
#include "src/algorithms/implicit_als/implicit_als_train_init_parameter.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResultBase, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_BASE_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);

PartialResultBase::PartialResultBase(size_t nElements) : daal::algorithms::PartialResult(nElements) {}

KeyValueDataCollectionPtr PartialResultBase::get(PartialResultBaseId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr PartialResultBase::get(PartialResultBaseId id, size_t key) const
{
    KeyValueDataCollectionPtr collection = get(id);
    NumericTablePtr nt;
    if (collection)
    {
        nt = NumericTable::cast((*collection)[key]);
    }
    return nt;
}

void PartialResultBase::set(PartialResultBaseId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status PartialResultBase::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);
    return Status();
}

Status PartialResultBase::checkImpl(size_t nParts) const
{
    KeyValueDataCollectionPtr collection = get(offsets);
    DAAL_CHECK_EX(collection.get(), ErrorNullPartialResult, ArgumentName, offsetsStr());
    DAAL_CHECK_EX(collection->size() == nParts, ErrorIncorrectDataCollectionSize, ArgumentName, offsetsStr());

    Status s;
    int unexpectedLayouts = (int)packed_mask;
    for (size_t i = 0; i < nParts; i++)
    {
        NumericTable * nt = NumericTable::cast((*collection)[i]).get();
        DAAL_CHECK_STATUS(s, checkNumericTable(nt, offsetsStr(), unexpectedLayouts, 0, 1, 1));
    }

    collection = get(outputOfInitForComputeStep3);
    DAAL_CHECK_EX(collection.get(), ErrorNullPartialResult, ArgumentName, outputOfInitForComputeStep3Str());
    DAAL_CHECK_EX(collection->size() == nParts, ErrorIncorrectDataCollectionSize, ArgumentName, outputOfInitForComputeStep3Str());
    for (size_t i = 0; i < nParts; i++)
    {
        NumericTable * nt = NumericTable::cast((*collection)[i]).get();
        DAAL_CHECK_STATUS(s, checkNumericTable(nt, outputOfInitForComputeStep3Str(), unexpectedLayouts, 0, 1, 0, false));
    }
    return s;
}

PartialResult::PartialResult() : PartialResultBase(lastPartialResultCollectionId + 1) {}

PartialModelPtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<PartialModel, SerializationIface>(Argument::get(id));
}

void PartialResult::set(PartialResultId id, const PartialModelPtr & ptr)
{
    Argument::set(id, ptr);
}

KeyValueDataCollectionPtr PartialResult::get(PartialResultCollectionId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr PartialResult::get(PartialResultCollectionId id, size_t key) const
{
    KeyValueDataCollectionPtr collection = get(id);
    NumericTablePtr nt;
    if (collection)
    {
        nt = NumericTable::cast((*collection)[key]);
    }
    return nt;
}

void PartialResult::set(PartialResultCollectionId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s = PartialResultBase::check(input, parameter, method);
    if (!s) return s;

    const DistributedParameter * algParameter           = static_cast<const DistributedParameter *>(parameter);
    SharedPtr<HomogenNumericTable<int> > partitionTable = internal::getPartition(algParameter, s);
    if (!s) return s;

    size_t nParts = partitionTable->getNumberOfRows() - 1;
    DAAL_CHECK_STATUS(s, checkImpl(nParts));

    const DistributedInput<step1Local> * algInput = static_cast<const DistributedInput<step1Local> *>(input);
    size_t nRows                                  = algInput->get(data)->getNumberOfRows();
    size_t nCols                                  = algInput->get(data)->getNumberOfColumns();
    DAAL_CHECK_EX(algParameter->fullNUsers >= nCols, ErrorIncorrectParameter, ParameterName, fullNUsersStr());

    PartialModelPtr model = get(partialModel);
    DAAL_CHECK(model, ErrorNullPartialModel);

    size_t nFactors       = algParameter->nFactors;
    int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayouts, 0, nFactors, nRows));
    DAAL_CHECK_STATUS(s, checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayouts, 0, 1, nRows));

    KeyValueDataCollectionPtr collection = get(outputOfStep1ForStep2);
    DAAL_CHECK_EX(collection.get(), ErrorNullPartialResult, ArgumentName, outputOfStep1ForStep2Str());
    DAAL_CHECK_EX(collection->size() == nParts, ErrorIncorrectDataCollectionSize, ArgumentName, outputOfStep1ForStep2Str());

    int * partitionData = partitionTable->getArray();
    int expectedLayout  = (int)NumericTableIface::csrArray;

    NumericTable * nt = NumericTable::cast((*collection)[0]).get();
    size_t resNRows   = partitionData[1] - partitionData[0];
    DAAL_CHECK_STATUS(s, checkNumericTable(nt, outputOfStep1ForStep2Str(), 0, expectedLayout, 0, resNRows, false));
    size_t resNCols = nt->getNumberOfColumns();
    DAAL_CHECK_EX(resNCols == nRows, ErrorIncorrectNumberOfColumns, ArgumentName, outputOfStep1ForStep2Str());

    for (size_t i = 1; i < nParts; i++)
    {
        nt       = NumericTable::cast((*collection)[i]).get();
        resNRows = partitionData[i + 1] - partitionData[i];
        DAAL_CHECK_STATUS(s, checkNumericTable(nt, outputOfStep1ForStep2Str(), 0, expectedLayout, resNCols, resNRows, false));
    }
    return s;
}

DistributedPartialResultStep2::DistributedPartialResultStep2() : PartialResultBase(lastDistributedPartialResultStep2Id + 1) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep2::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s                                      = PartialResultBase::check(input, parameter, method);
    const DistributedInput<step2Local> * algInput = static_cast<const DistributedInput<step2Local> *>(input);

    KeyValueDataCollectionPtr collection = algInput->get(inputOfStep2FromStep1);
    const size_t nParts                  = collection->size();
    DAAL_CHECK_STATUS(s, checkImpl(nParts));

    size_t nRows = NumericTable::cast((*collection)[0])->getNumberOfRows();
    size_t nCols = 0;
    for (size_t i = 0; i < nParts; i++)
    {
        nCols += NumericTable::cast((*collection)[i])->getNumberOfColumns();
    }

    const int expectedLayout = (int)NumericTableIface::csrArray;
    return checkNumericTable(get(transposedData).get(), transposedDataStr(), 0, expectedLayout, nCols, nRows, false);
}

} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
