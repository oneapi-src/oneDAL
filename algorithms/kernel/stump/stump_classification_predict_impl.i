/* file: stump_classification_predict_impl.i */
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

#ifndef __STUMP_CLASSIFICATION_PREDICT_IMPL_I__
#define __STUMP_CLASSIFICATION_PREDICT_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "daal_defines.h"
#include "daal_shared_ptr.h"
#include "service_numeric_table.h"
#include "decision_tree_classification_predict.h"
#include "decision_tree_classification_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
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
services::Status StumpPredictKernel<method, algorithmFPtype, cpu>::changeZeroToMinusOne(NumericTable * yTable)
{
    services::Status s;
    const size_t nVectors = yTable->getNumberOfRows();
    WriteColumns<algorithmFPtype, cpu> y(yTable, 0, 0, nVectors);
    DAAL_CHECK_STATUS(s, y.status());
    algorithmFPtype * yArray = y.get();
    DAAL_CHECK(yArray, ErrorMemoryAllocationFailed);
    const algorithmFPtype zero     = 0.0;
    const algorithmFPtype minusOne = -1.0;
    for (size_t i = 0; i < nVectors; i++)
    {
        if (yArray[i] == zero)
        {
            yArray[i] = minusOne;
        }
    }
    return s;
}

template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpPredictKernel<method, algorithmFPtype, cpu>::compute(const NumericTable * xTable, const stump::classification::Model * m,
                                                                           NumericTable * rTableLabels, NumericTable * rTableProb,
                                                                           const Parameter * par)
{
    services::Status s;
    const size_t nClasses = par->nClasses;

    decision_tree::classification::prediction::Batch<algorithmFPtype> treeAlgorithm(nClasses);
    treeAlgorithm.enableChecks(false);

    treeAlgorithm.input.set(classifier::prediction::data, NumericTablePtr(const_cast<NumericTable *>(xTable), EmptyDeleter()));
    treeAlgorithm.input.set(classifier::prediction::model,
                            decision_tree::classification::ModelPtr(
                                static_cast<decision_tree::classification::Model *>(const_cast<stump::classification::Model *>(m)), EmptyDeleter()));
    treeAlgorithm.parameter.resultsToEvaluate = par->resultsToEvaluate;
    classifier::prediction::ResultPtr treeResult(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(treeResult.get())
    treeResult->set(daal::algorithms::classifier::prediction::prediction, NumericTablePtr(rTableLabels, EmptyDeleter()));
    treeResult->set(daal::algorithms::classifier::prediction::probabilities, NumericTablePtr(rTableProb, EmptyDeleter()));
    treeAlgorithm.setResult(treeResult);

    DAAL_CHECK_STATUS(s, treeAlgorithm.computeNoThrow());

    if (rTableLabels && nClasses == 2)
    {
        DAAL_CHECK_STATUS(s, changeZeroToMinusOne(rTableLabels));
    }

    return s;
}

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
