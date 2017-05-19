/* file: df_classification_training_result.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the decision forest algorithm interface
//--
*/

#ifndef __DF_CLASSIFICATION_TRAINING_RESULT_H
#define __DF_CLASSIFICATION_TRAINING_RESULT_H

#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "df_classification_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{

/**
 * Allocates memory to store the result of decision forest model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] parameter %Parameter of decision forest model-based training
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *prm, const int method)
{
    const Parameter *parameter = static_cast<const Parameter *>(prm);

    set(classifier::training::model, daal::algorithms::decision_forest::classification::ModelPtr(new decision_forest::classification::internal::ModelImpl()));
    if(parameter->resultsToCompute & decision_forest::training::computeOutOfBagError)
    {
        set(outOfBagError, NumericTablePtr(new data_management::HomogenNumericTable<algorithmFPType>(1, 1,
            data_management::NumericTable::doAllocate)));
    }
    if(parameter->varImportance != decision_forest::training::none)
    {
        const classifier::training::Input *inp = static_cast<const classifier::training::Input *>(input);
        const size_t nFeatures = inp->get(classifier::training::data)->getNumberOfColumns();
        set(variableImportance, NumericTablePtr(new data_management::HomogenNumericTable<algorithmFPType>(nFeatures,
            1, data_management::NumericTable::doAllocate)));
    }
    return services::Status();
}

} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
