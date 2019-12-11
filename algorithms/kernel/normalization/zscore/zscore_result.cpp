/* file: zscore_result.cpp */
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
//  Implementation of zscore internal result.
//--
*/

#include "zscore_result.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "service_defines.h"

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
namespace interface2
{
/**
* Checks the correctness of the Result object
* \param[in] in     Pointer to the input object
* \param[in] par    Pointer to the parameter object
*/
Status ResultImpl::check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par) const
{
    Status status = interface1::ResultImpl::check(in);
    DAAL_CHECK_STATUS_VAR(status);

    const interface1::Input * input = static_cast<const interface1::Input *>(in);
    DAAL_CHECK(input, ErrorNullInput);

    NumericTablePtr dataTable = input->get(zscore::data);
    DAAL_CHECK(dataTable, ErrorNullInputNumericTable);

    const size_t nFeatures = dataTable->getNumberOfColumns();

    const interface3::BaseParameter * parameter = static_cast<const BaseParameter *>(par);
    if (parameter->resultsToCompute & mean)
    {
        DAAL_CHECK_STATUS(status, checkNumericTable(NumericTable::cast(get(means)).get(), meansStr(), packed_mask, 0, nFeatures, 1))
    }
    if (parameter->resultsToCompute & variance)
    {
        DAAL_CHECK_STATUS(status, checkNumericTable(NumericTable::cast(get(variances)).get(), variancesStr(), packed_mask, 0, nFeatures, 1))
    }
    return status;
}

} // namespace interface2

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
