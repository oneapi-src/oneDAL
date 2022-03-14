/* file: covariance_dense_distr_step2_oneapi.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Declaration of template structs that calculate Covariance matrix.
//--
*/

#ifndef __COVARIANCE_DENSE_DISTR_STEP2_ONEAPI_H__
#define __COVARIANCE_DENSE_DISTR_STEP2_ONEAPI_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/covariance/covariance_types.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace oneapi
{
namespace internal
{
template <typename algorithmFPType, Method method>
class CovarianceDenseDistrStep2KernelOneAPI : public Kernel
{
public:
    services::Status compute(DataCollection * partialResultsCollection, NumericTable * nObsTable, NumericTable * crossProductTable,
                             NumericTable * sumTable, const Parameter * parameter);

    services::Status finalizeCompute(NumericTable * nObsTable, NumericTable * crossProductTable, NumericTable * sumTable, NumericTable * covTable,
                                     NumericTable * meanTable, const Parameter * parameter);
};

} // namespace internal
} // namespace oneapi
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
