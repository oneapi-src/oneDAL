/* file: gbt_classification_training_result_fpt.cpp */
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

#include "algorithms/gradient_boosted_trees/gbt_classification_training_types.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status s;
    const classifier::training::Input * inp = static_cast<const classifier::training::Input *>(input);
    const size_t nFeatures                  = inp->get(classifier::training::data)->getNumberOfColumns();
    set(classifier::training::model, daal::algorithms::gbt::classification::Model::create(nFeatures, &s));

    const Parameter * par = dynamic_cast<const Parameter *>(parameter);
    if (par != nullptr)
    {
        if (par->varImportance & gbt::training::gain)
        {
            set(variableImportanceByGain,
                data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
        }
        if (par->varImportance & gbt::training::totalGain)
        {
            set(variableImportanceByTotalGain,
                data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
        }
        if (par->varImportance & gbt::training::cover)
        {
            set(variableImportanceByCover,
                data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
        }
        if (par->varImportance & gbt::training::totalCover)
        {
            set(variableImportanceByTotalCover,
                data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
        }
        if (par->varImportance & gbt::training::weight)
        {
            set(variableImportanceByWeight,
                data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, data_management::NumericTable::doAllocate, 0, &s));
        }
    }

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace training
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
