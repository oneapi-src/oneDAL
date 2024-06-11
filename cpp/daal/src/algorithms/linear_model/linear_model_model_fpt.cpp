/* file: linear_model_model_fpt.cpp */
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

#include "src/algorithms/linear_model/linear_model_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace internal
{
using namespace daal::data_management;

template <typename modelFPType>
ModelInternal::ModelInternal(size_t nFeatures, size_t nResponses, const Parameter & par, modelFPType dummy) : _interceptFlag(par.interceptFlag)
{
    services::Status st;

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        _beta = HomogenNumericTable<modelFPType>::create(nFeatures + 1, nResponses, NumericTable::doAllocate, 0, &st);
    }
    if (!st) return;
}

template ModelInternal::ModelInternal(size_t nFeatures, size_t nResponses, const Parameter & par, DAAL_FPTYPE dummy);
} // namespace internal
} // namespace linear_model
} // namespace algorithms
} // namespace daal
