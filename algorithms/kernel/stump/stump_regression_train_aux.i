/* file: stump_regression_train_aux.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_REGRESSION_TRAIN_AUX_I__
#define __STUMP_REGRESSION_TRAIN_AUX_I__

#include "daal_defines.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "decision_tree_model.h"
#include "decision_tree_regression_training_batch.h"
#include "regression_training_types.h"
#include "stump_regression_model.h"
#include "decision_tree_regression_model_impl.h"

using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
namespace training
{
namespace internal
{
/**
 *  \brief Perform stump regression for data set X on responses Y with weights W
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpTrainKernel<method, algorithmFPtype, cpu>::compute(size_t n, const NumericTable * const * a,
                                                                         stump::regression::Model * stumpModel, const Parameter * par)
{
    const NumericTable * xTable = a[0];
    NumericTable * yTable       = const_cast<NumericTable *>(a[1]);
    const NumericTable * wTable = (n >= 3 ? a[2] : 0);

    services::Status s;

    /* Create an algorithm object to train the Decision tree model */
    decision_tree::regression::training::Batch<> treeAlgorithm;
    treeAlgorithm.enableChecks(false);
    treeAlgorithm.parameter.pruning                    = decision_tree::none;
    treeAlgorithm.parameter.maxTreeDepth               = 2;
    treeAlgorithm.parameter.minObservationsInLeafNodes = 1;

    /* Pass the training data set, labels, and pruning dataset with labels to the algorithm */
    treeAlgorithm.input.set(decision_tree::regression::training::data, NumericTablePtr(const_cast<NumericTable *>(xTable), EmptyDeleter()));
    treeAlgorithm.input.set(decision_tree::regression::training::weights, NumericTablePtr(const_cast<NumericTable *>(wTable), EmptyDeleter()));
    treeAlgorithm.input.set(decision_tree::regression::training::dependentVariables,
                            NumericTablePtr(const_cast<NumericTable *>(yTable), EmptyDeleter()));

    decision_tree::regression::training::ResultPtr treeResult(new decision_tree::regression::training::Result());
    treeResult->set(algorithms::regression::training::model,
                    decision_tree::regression::ModelPtr(
                        static_cast<decision_tree::regression::Model *>(const_cast<stump::regression::Model *>(stumpModel)), EmptyDeleter()));
    treeAlgorithm.setResult(treeResult);
    DAAL_CHECK_STATUS(s, treeAlgorithm.computeNoThrow());

    return s;
}

} // namespace internal
} // namespace training
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
