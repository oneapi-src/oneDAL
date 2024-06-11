/* file: linear_model_predict_batch_fpt.cpp */
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
//  Implementation of the regression algorithm interface
//--
*/

#include "algorithms/linear_model/linear_model_predict_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
using namespace daal::services;
using namespace daal::data_management;

template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const Input * in           = static_cast<const Input *>(input);
    size_t nVectors            = in->get(data)->getNumberOfRows();
    size_t nDependentVariables = in->get(model)->getNumberOfResponses();
    Status st;

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        set(prediction, HomogenNumericTable<algorithmFPType>::create(nDependentVariables, nVectors, NumericTable::doAllocate, &st));
    }
    return st;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                          const int method);

} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal
