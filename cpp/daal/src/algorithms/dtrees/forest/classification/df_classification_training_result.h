/* file: df_classification_training_result.h */
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

#ifndef __DF_CLASSIFICATION_TRAINING_RESULT_H__
#define __DF_CLASSIFICATION_TRAINING_RESULT_H__

#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
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
 * \param[in] input     Pointer to an object containing the input data
 * \param[in] method    Computation method for the algorithm
 * \param[in] parameter %Parameter of decision forest model-based training
 * \return Status of allocation
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * prm, const int method)
{
    services::Status status;
    const daal::algorithms::decision_forest::training::Parameter * parameter2 =
        dynamic_cast<const daal::algorithms::decision_forest::training::Parameter *>(prm);
    const classifier::training::Input * inp = static_cast<const classifier::training::Input *>(input);
    const size_t nFeatures                  = inp->get(classifier::training::data)->getNumberOfColumns();

    set(classifier::training::model,
        daal::algorithms::decision_forest::classification::ModelPtr(new decision_forest::classification::internal::ModelImpl(nFeatures)));
    if (parameter2 != NULL)
    {
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagError)
        {
            set(outOfBagError, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                   1, 1, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorAccuracy)
        {
            set(outOfBagErrorAccuracy, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                           1, 1, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
        {
            const size_t nObs = inp->get(classifier::training::data)->getNumberOfRows();
            set(outOfBagErrorPerObservation, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                                 1, nObs, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorDecisionFunction)
        {
            const decision_forest::classification::training::Parameter * parameter3 =
                dynamic_cast<const decision_forest::classification::training::Parameter *>(prm);
            const size_t nClasses = parameter3->nClasses;
            const size_t nObs     = inp->get(classifier::training::data)->getNumberOfRows();
            set(outOfBagErrorDecisionFunction, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                                   nClasses, nObs, data_management::NumericTable::doAllocate, status)));
        }
        if (parameter2->varImportance != decision_forest::training::none)
        {
            const classifier::training::Input * inp = static_cast<const classifier::training::Input *>(input);
            const size_t nFeatures                  = inp->get(classifier::training::data)->getNumberOfColumns();
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
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
