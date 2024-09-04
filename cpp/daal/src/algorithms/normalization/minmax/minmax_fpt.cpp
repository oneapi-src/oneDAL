/* file: minmax_fpt.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "algorithms/normalization/minmax_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{
/**
 * Allocates memory to store the result of the minmax normalization algorithm
 * \param[in] input  %Input object for the minmax normalization algorithm
 * \param[in] par    %Parameter of the minmax normalization algorithm
 * \param[in] method Computation method of the minmax normalization algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, int method)
{
    DAAL_CHECK(input, ErrorNullInput);

    const Input * algInput    = static_cast<const Input *>(input);
    NumericTablePtr dataTable = algInput->get(data);

    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const size_t nRows                  = dataTable->getNumberOfRows();
    const size_t nColumns               = dataTable->getNumberOfColumns();
    NumericTablePtr normalizedDataTable = HomogenNumericTable<algorithmFPType>::create(nColumns, nRows, NumericTable::doAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);
    set(normalizedData, normalizedDataTable);
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, int method);

} // namespace interface1
} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
