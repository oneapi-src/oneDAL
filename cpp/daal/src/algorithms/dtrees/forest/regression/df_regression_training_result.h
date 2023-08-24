/* file: df_regression_training_result.h */
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
//  Implementation of the decision forest algorithm interface
//--
*/

#ifndef __DF_REGRESSION_TRAINING_RESULT_H__
#define __DF_REGRESSION_TRAINING_RESULT_H__

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
/**
 * Allocates memory to store the result of decision forest model-based training
 * \param[in] input       %Input object for the algorithm
 * \param[in] method      Computation method for the algorithm
 * \param[in] parameter   %Parameters of the algorithm
 * \return Status of allocation
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status status;
    const daal::algorithms::decision_forest::training::Parameter * parameter2 =
        dynamic_cast<const daal::algorithms::decision_forest::training::Parameter *>(parameter);
    const Input * inp      = static_cast<const Input *>(input);
    const size_t nFeatures = inp->get(data)->getNumberOfColumns();
    set(model, daal::algorithms::decision_forest::regression::ModelPtr(new decision_forest::regression::internal::ModelImpl(nFeatures)));
    if (parameter2 != NULL)
    {
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagError)
        {
            set(outOfBagError, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                   1, 1, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorR2)
        {
            set(outOfBagErrorR2, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                     1, 1, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
        {
            const size_t nObs = inp->get(data)->getNumberOfRows();
            set(outOfBagErrorPerObservation, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                                 1, nObs, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorPrediction)
        {
            const size_t nObs = inp->get(data)->getNumberOfRows();
            set(outOfBagErrorPrediction, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                             1, nObs, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->varImportance != decision_forest::training::none)
        {
            set(variableImportance, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                        nFeatures, 1, data_management::NumericTable::doAllocate, status)));
        }
    }
    else
    {
        status = status ? status : services::Status(services::ErrorNullParameterNotSupported);
    }
    return status;
}

} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
