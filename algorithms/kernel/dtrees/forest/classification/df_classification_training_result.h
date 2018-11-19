/* file: df_classification_training_result.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
    services::Status status;
    const Parameter *parameter = static_cast<const Parameter *>(prm);
    const classifier::training::Input* inp = static_cast<const classifier::training::Input*>(input);
    const size_t nFeatures = inp->get(classifier::training::data)->getNumberOfColumns();

    set(classifier::training::model, daal::algorithms::decision_forest::classification::ModelPtr(new decision_forest::classification::internal::ModelImpl(nFeatures)));
    if(parameter->resultsToCompute & decision_forest::training::computeOutOfBagError)
    {
        set(outOfBagError, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(1, 1,
            data_management::NumericTable::doAllocate, status)));
    }
    if(parameter->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
    {
        const size_t nObs = inp->get(classifier::training::data)->getNumberOfRows();
        set(outOfBagErrorPerObservation, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(1, nObs,
            data_management::NumericTable::doAllocate, status)));
    }
    if(parameter->varImportance != decision_forest::training::none)
    {
        const classifier::training::Input *inp = static_cast<const classifier::training::Input *>(input);
        const size_t nFeatures = inp->get(classifier::training::data)->getNumberOfColumns();
        set(variableImportance, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures,
            1, data_management::NumericTable::doAllocate, status)));
    }
    return status;
}

} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
