/* file: zscore_result_v1_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "zscore_result_v1.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface1
{
/**
* Allocates memory to store final results of the z-score normalization algorithms
* \param[in] input     Input objects for the z-score normalization algorithm
* \param[in] method    Algorithm computation method
*/
template <typename algorithmFPType>
Status ResultImpl::allocate(const daal::algorithms::Input * input)
{
    const Input * in = static_cast<const Input *>(input);
    DAAL_CHECK(in, ErrorNullInput);

    NumericTablePtr dataTable = in->get(zscore::data);
    DAAL_CHECK(dataTable, ErrorNullInputNumericTable);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    const size_t nVectors  = dataTable->getNumberOfRows();

    Status status;
    (*this)[normalizedData] = HomogenNumericTable<algorithmFPType>::create(nFeatures, nVectors, NumericTable::doAllocate, &status);
    return status;
}

template DAAL_EXPORT Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input);

} // namespace interface1

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
