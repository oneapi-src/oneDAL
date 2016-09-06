/* file: binary_confusion_matrix_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Declaration of template class that computes binary confusion matrix.
//--
*/

#ifndef __BINARY_CONFUSION_MATRIX_DEFAULT_IMPL_I__
#define __BINARY_CONFUSION_MATRIX_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_micro_table.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace quality_metric
{
namespace binary_confusion_matrix
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
void BinaryConfusionMatrixKernel<method, algorithmFPType, cpu>::compute(const NumericTable *predictedLabelsTable,
                                                                        const NumericTable *groundTruthLabelsTable,
                                                                        NumericTable *confusionMatrixTable,
                                                                        NumericTable *accuracyMeasuresTable,
                                                                        const binary_confusion_matrix::Parameter *parameter)
{
    const algorithmFPType zero = 0.0;

    FeatureMicroTable<algorithmFPType, readOnly, cpu> mtPredictedLabels(predictedLabelsTable);
    FeatureMicroTable<algorithmFPType, readOnly, cpu> mtGroundTruthLabels(groundTruthLabelsTable);

    BlockMicroTable<int,             writeOnly, cpu> mtConfusionMatrix(confusionMatrixTable);
    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtAccuracyMeasures(accuracyMeasuresTable);

    algorithmFPType *predictedLabelsData, *groundTruthLabelsData;
    int *confusionMatrixData;
    algorithmFPType *accuracyMeasuresData;

    size_t nVectors = predictedLabelsTable->getNumberOfRows();
    size_t nVectorsReturned = 0;
    algorithmFPType fpNVectors = (algorithmFPType)nVectors;
    algorithmFPType invNVectors = 1.0 / fpNVectors;

    /* Get input data */
    nVectorsReturned = mtPredictedLabels.getBlockOfColumnValues(0, 0, nVectors, &predictedLabelsData);
    if (nVectorsReturned < nVectors) { this->_errors->add(ErrorIncorrectNumberOfObservations); return; }

    nVectorsReturned = mtGroundTruthLabels.getBlockOfColumnValues(0, 0, nVectors, &groundTruthLabelsData);
    if (nVectorsReturned < nVectors)
    {
        mtPredictedLabels.release();
        this->_errors->add(ErrorIncorrectNumberOfObservations);
        return;
    }

    algorithmFPType beta = parameter->beta;
    algorithmFPType beta2 = beta * beta;
    size_t nClasses = 0;

    nClasses = mtConfusionMatrix.getBlockOfRows(0, 2, &confusionMatrixData);
    if (nClasses < 2)
    {
        mtPredictedLabels.release();
        mtGroundTruthLabels.release();
        this->_errors->add(ErrorIncorrectNumberOfObservations); return;
    }

    size_t nRowsReturned = mtAccuracyMeasures.getBlockOfRows(0, 1, &accuracyMeasuresData);
    if (nRowsReturned < 1)
    {
        mtPredictedLabels.release();
        mtGroundTruthLabels.release();
        mtConfusionMatrix.release();
        this->_errors->add(ErrorIncorrectNumberOfObservations); return;
    }

    service_memset<int, cpu>(confusionMatrixData, 0, nClasses * nClasses);

    /* Compute confusion matrix for two-class classifier */
    for (size_t i = 0; i < nVectors; i++)
    {
        int predictedLabel   = ((predictedLabelsData[i]   > zero) ? 0 : 1);
        int groundTruthLabel = ((groundTruthLabelsData[i] > zero) ? 0 : 1);

        confusionMatrixData[groundTruthLabel * 2 + predictedLabel]++;
    }
    algorithmFPType tp, fp, tn, fn;

    tp = (algorithmFPType)confusionMatrixData[0];
    fn = (algorithmFPType)confusionMatrixData[1];
    fp = (algorithmFPType)confusionMatrixData[2];
    tn = (algorithmFPType)confusionMatrixData[3];

    /* Accuracy */
    accuracyMeasuresData[0] = (tp + tn) / fpNVectors;
    /* Precision */
    accuracyMeasuresData[1] = tp / (tp + fp);
    /* Recall */
    accuracyMeasuresData[2] = tp / (tp + fn);
    /* F-score */
    accuracyMeasuresData[3] = ((beta2 + 1.0)*tp)/((beta2 + 1.0)*tp + beta2*fn + fp);
    /* Specificity */
    accuracyMeasuresData[4] = tn / (fp + tn);
    /* AUC (ability to avoid false classification) */
    accuracyMeasuresData[5] = 0.5 * (accuracyMeasuresData[2] + accuracyMeasuresData[4]);

    mtPredictedLabels.release();
    mtGroundTruthLabels.release();
    mtConfusionMatrix.release();
    mtAccuracyMeasures.release();
}

}
}
}
}
}
}

#endif
