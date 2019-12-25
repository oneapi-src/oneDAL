/* file: stump_regression_predict_impl.i */
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

#ifndef __STUMP_REGRESSION_PREDICT_IMPL_I__
#define __STUMP_REGRESSION_PREDICT_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "daal_defines.h"
#include "daal_shared_ptr.h"
#include "service_numeric_table.h"
#include "algorithms/decision_tree/decision_tree_regression_predict.h"
#include "algorithms/decision_tree/decision_tree_regression_predict_types.h"
#include "algorithms/stump/stump_regression_model.h"
#include "decision_tree_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::services;

template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpPredictKernel<method, algorithmFPtype, cpu>::compute(const NumericTable * xTable, const stump::regression::Model * m,
                                                                           NumericTable * rTable, const Parameter * par)
{
    services::Status s;
    decision_tree::regression::prediction::Batch<> treeAlgorithm;
    treeAlgorithm.enableChecks(false);

    treeAlgorithm.input.set(daal::algorithms::decision_tree::regression::prediction::data,
                            NumericTablePtr(const_cast<NumericTable *>(xTable), EmptyDeleter()));
    treeAlgorithm.input.set(daal::algorithms::decision_tree::regression::prediction::model,
                            decision_tree::regression::ModelPtr(
                                static_cast<decision_tree::regression::Model *>(const_cast<stump::regression::Model *>(m)), EmptyDeleter()));
    decision_tree::regression::prediction::ResultPtr treeResult(new decision_tree::regression::prediction::Result());
    treeResult->set(decision_tree::regression::prediction::prediction, NumericTablePtr(rTable, EmptyDeleter()));
    treeAlgorithm.setResult(treeResult);

    DAAL_CHECK_STATUS(s, treeAlgorithm.computeNoThrow());

    return s;
}

} // namespace internal
} // namespace prediction
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
