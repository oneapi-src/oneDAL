/* file: stump_classification_train_aux.i */
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
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_CLASSIFICATION_TRAIN_AUX_I__
#define __STUMP_CLASSIFICATION_TRAIN_AUX_I__

#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "algorithms/decision_tree/decision_tree_model.h"
#include "algorithms/decision_tree/decision_tree_classification_training_batch.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "services/daal_memory.h"

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
namespace classification
{
namespace training
{
namespace internal
{
template <Method method, typename algorithmFPtype, CpuType cpu>
Status StumpTrainKernel<method, algorithmFPtype, cpu>::changeMinusOneToZero(const algorithmFPtype * yArray, algorithmFPtype * yZeroOne,
                                                                            size_t nVectors)
{
    const algorithmFPtype one      = 1.0;
    const algorithmFPtype zero     = 0.0;
    const algorithmFPtype minusOne = -1.0;
    for (size_t i = 0; i < nVectors; i++)
    {
        // exact fp comparison, fp stores labels: exact numbers 1 or -1
        if (yArray[i] == minusOne)
        {
            yZeroOne[i] = zero;
        }
        else
        {
            yZeroOne[i] = one;
        }
    }
    return Status();
}

/**
 *  \brief Perform stump regression for data set X on responses Y with weights W
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
Status StumpTrainKernel<method, algorithmFPtype, cpu>::compute(size_t n, const NumericTable * const * a, stump::classification::Model * stumpModel,
                                                               const Parameter * par)
{
    Status s;

    const NumericTable * xTable = a[0];
    NumericTable * yTable       = const_cast<NumericTable *>(a[1]);
    const NumericTable * wTable = (n >= 3 ? a[2] : 0);

    const size_t nFeatures = xTable->getNumberOfColumns();
    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nClasses  = par->nClasses;
    stumpModel->setNFeatures(nFeatures);

    SharedPtr<HomogenNumericTableCPU<algorithmFPtype, cpu> > yTableZeroOne;
    if (nClasses == 2)
    {
        yTableZeroOne = HomogenNumericTableCPU<algorithmFPtype, cpu>::create(1, nVectors, &s);
        DAAL_CHECK_STATUS_VAR(s);
        algorithmFPtype * yZeroOne = yTableZeroOne->getArray();

        ReadColumns<algorithmFPtype, cpu> y(yTable, 0, 0, nVectors);
        DAAL_CHECK_STATUS(s, y.status());
        const algorithmFPtype * yArray = y.get();

        if (yZeroOne && yArray)
        {
            int result = 0;
            result =
                daal::services::internal::daal_memcpy_s(yZeroOne, nVectors * sizeof(algorithmFPtype), yArray, nVectors * sizeof(algorithmFPtype));
            if (result)
            {
                s |= Status(ErrorMemoryCopyFailedInternal);
            }
            DAAL_CHECK_STATUS(s, changeMinusOneToZero(yArray, yZeroOne, nVectors));
        }
        yTable = yTableZeroOne.get();
    }

    /* Create an algorithm object to train the Decision tree model */
    decision_tree::classification::training::Batch<algorithmFPtype> treeAlgorithm(par->nClasses);
    treeAlgorithm.enableChecks(false);
    treeAlgorithm.parameter.splitCriterion             = par->splitCriterion;
    treeAlgorithm.parameter.pruning                    = decision_tree::none;
    treeAlgorithm.parameter.maxTreeDepth               = 2;
    treeAlgorithm.parameter.minObservationsInLeafNodes = 1;

    /* Pass the training data set, labels, and pruning dataset with labels to the algorithm */
    treeAlgorithm.input.set(classifier::training::data, NumericTablePtr(const_cast<NumericTable *>(xTable), EmptyDeleter()));
    treeAlgorithm.input.set(classifier::training::weights, NumericTablePtr(const_cast<NumericTable *>(wTable), EmptyDeleter()));
    treeAlgorithm.input.set(classifier::training::labels, NumericTablePtr(const_cast<NumericTable *>(yTable), EmptyDeleter()));

    decision_tree::classification::training::ResultPtr treeResult(new decision_tree::classification::training::Result());
    DAAL_CHECK_MALLOC(treeResult.get())
    treeResult->set(classifier::training::model,
                    decision_tree::classification::ModelPtr(
                        static_cast<decision_tree::classification::Model *>(const_cast<stump::classification::Model *>(stumpModel)), EmptyDeleter()));
    treeAlgorithm.setResult(treeResult);
    /* Train the Decision tree model */
    DAAL_CHECK_STATUS(s, treeAlgorithm.computeNoThrow());
    return s;
}

} // namespace internal
} // namespace training
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
