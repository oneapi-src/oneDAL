/* file: gbt_regression_training_result_fpt.cpp */
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
//  Implementation of the gradient boosted trees algorithm interface
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
using namespace daal::data_management;

template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method)
{
    services::Status s;
    const Input * inp      = static_cast<const Input *>(input);
    const size_t nFeatures = inp->get(data)->getNumberOfColumns();
    set(model, daal::algorithms::gbt::regression::Model::create(nFeatures, &s));

    const Parameter * par = static_cast<const Parameter *>(parameter);
    if (par->varImportance & gbt::training::weight)
    {
        set(variableImportanceByWeight,
            data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
    }
    if (par->varImportance & gbt::training::totalCover)
    {
        set(variableImportanceByTotalCover,
            data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
    }
    if (par->varImportance & gbt::training::cover)
    {
        set(variableImportanceByCover,
            data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
    }
    if (par->varImportance & gbt::training::totalGain)
    {
        set(variableImportanceByTotalGain,
            data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
    }
    if (par->varImportance & gbt::training::gain)
    {
        set(variableImportanceByGain,
            data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
    }

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const Parameter * parameter,
                                                                    const int method);

} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
