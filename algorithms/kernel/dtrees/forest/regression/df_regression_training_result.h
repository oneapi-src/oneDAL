/* file: df_regression_training_result.h */
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

#ifndef __DF_REGRESSION_TRAINING_RESULT_
#define __DF_REGRESSION_TRAINING_RESULT_

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "df_regression_model_impl.h"

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
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] parameter %Parameter of decision forest model-based training
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const Parameter *parameter, const int method)
{
    services::Status status;
    const Input* inp = static_cast<const Input*>(input);
    const size_t nFeatures = inp->get(data)->getNumberOfColumns();
    set(model, daal::algorithms::decision_forest::regression::ModelPtr(new decision_forest::regression::internal::ModelImpl(nFeatures)));
    if(parameter->resultsToCompute & decision_forest::training::computeOutOfBagError)
    {
        set(outOfBagError, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(1, 1,
            data_management::NumericTable::doAllocate, status)));
    }
    if(parameter->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
    {
        const size_t nObs = inp->get(data)->getNumberOfRows();
        set(outOfBagErrorPerObservation, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(1, nObs,
            data_management::NumericTable::doAllocate, status)));
    }
    if(parameter->varImportance != decision_forest::training::none)
    {
        set(variableImportance, NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures,
            1, data_management::NumericTable::doAllocate, status)));
    }
    return status;
}

} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
