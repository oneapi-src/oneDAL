/* file: zscore_v1.cpp */
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

#include "zscore_types_v1.h"
#include "zscore_result_v1.h"
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
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NORMALIZATION_ZSCORE_RESULT_ID);

Result::Result() : daal::algorithms::Result(lastResultId + 1)
{
    Argument::setStorage(data_management::DataCollectionPtr(new ResultImpl(lastResultId + 1)));
}

Result::Result(const Result & o)
{
    ResultImpl * pImpl = dynamic_cast<ResultImpl *>(getStorage(o).get());
    DAAL_ASSERT(pImpl);
    Argument::setStorage(data_management::DataCollectionPtr(new ResultImpl(*pImpl)));
}

/**
 * Returns the final result of the z-score normalization algorithm
 * \param[in] id   Identifier of the final result, daal::algorithms::normalization::zscore::ResultId
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets the Result object of the z-score normalization algorithm
 * \param[in] id        Identifier of the Result object
 * \param[in] value     Pointer to the Result object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] in     Pointer to the input object
 * \param[in] par    Pointer to the parameter object
 * \param[in] method Algorithm computation method
 */
Status Result::check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par, int method) const
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl.get(), ErrorNullPtr);
    return impl->check(in);
}

} // namespace interface1

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
