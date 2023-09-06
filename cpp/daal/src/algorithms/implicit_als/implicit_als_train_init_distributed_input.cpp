/* file: implicit_als_train_init_distributed_input.cpp */
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
#include "src/data_management/service_numeric_table.h"
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
namespace interface1
{
DistributedParameter::DistributedParameter(size_t nFactors, size_t fullNUsers, size_t seed) : Parameter(nFactors, fullNUsers, seed) {}

services::Status DistributedParameter::check() const
{
    services::Status s = Parameter::check();
    if (!s) return s;

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(partition.get(), partitionStr(), unexpectedLayouts, 0, 1));
    size_t nRows = partition->getNumberOfRows();
    DAAL_CHECK_EX(nRows > 0, ErrorIncorrectNumberOfRows, ParameterName, partitionStr());

    daal::internal::ReadRows<int, DAAL_BASE_CPU> partitionRows(partition.get(), 0, nRows);
    const int * p = partitionRows.get();
    DAAL_ASSERT(p);

    if (nRows == 1)
    {
        /* Here if the partition table of size 1x1 contains the number of parts in distributed data set */
        /* The number of parts should be greater than zero */
        DAAL_CHECK_EX(p[0] > 0, ErrorIncorrectParameter, ParameterName, partitionStr());
    }
    else
    {
        /* Here if the partition table of size nRows x 1 contains the offsets to each data part */
        /* Check that the offsets are stored in ascending order, first offset == 0 and the last element == fullNUsers */
        DAAL_CHECK_EX(p[0] == 0, ErrorIncorrectParameter, ParameterName, partitionStr());
        for (size_t i = 1; i < nRows; i++)
        {
            DAAL_CHECK_EX(p[i - 1] < p[i], ErrorIncorrectParameter, ParameterName, partitionStr());
        }
        DAAL_CHECK_EX(p[nRows - 1] == fullNUsers, ErrorIncorrectParameter, ParameterName, partitionStr());
    }

    return s;
}

DistributedInput<step1Local>::DistributedInput() : implicit_als::training::init::Input() {}

DistributedInput<step2Local>::DistributedInput() : daal::algorithms::Input(lastStep2LocalInputId + 1) {}

/**
 * Returns an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
KeyValueDataCollectionPtr DistributedInput<step2Local>::get(Step2LocalInputId id) const
{
    return KeyValueDataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Local>::set(Step2LocalInputId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of the implicit ALS initialization algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step2Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_EX(get(inputOfStep2FromStep1).get(), ErrorNullInputDataCollection, ArgumentName, inputOfStep2FromStep1Str());

    KeyValueDataCollection & collection = *(get(inputOfStep2FromStep1));
    size_t nParts                       = collection.size();
    DAAL_CHECK_EX(nParts, ErrorIncorrectDataCollectionSize, ArgumentName, inputOfStep2FromStep1Str());

    for (size_t i = 0; i < nParts; i++)
    {
        if (!NumericTable::cast(collection[i]).get())
            return Status(Error::create(ErrorIncorrectElementInNumericTableCollection, ArgumentName, inputOfStep2FromStep1Str()));
    }

    int expectedLayout = (int)NumericTableIface::csrArray;
    s                  = checkNumericTable(NumericTable::cast(collection[0]).get(), inputOfStep2FromStep1Str(), 0, expectedLayout);
    if (!s)
    {
        s.add(Error::create(ErrorIncorrectElementInNumericTableCollection, ElementInCollection, 0));
        return s;
    }

    size_t nRows = NumericTable::cast(collection[0])->getNumberOfRows();

    for (size_t i = 1; i < nParts; i++)
    {
        s = checkNumericTable(NumericTable::cast(collection[i]).get(), inputOfStep2FromStep1Str(), 0, expectedLayout, 0, nRows);
        if (!s)
        {
            s.add(Error::create(ErrorIncorrectElementInNumericTableCollection, ElementInCollection, i));
            return s;
        }
    }
    return services::Status();
}

} // namespace interface1
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
