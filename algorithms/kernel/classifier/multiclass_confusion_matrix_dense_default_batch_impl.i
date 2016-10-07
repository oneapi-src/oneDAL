/* file: multiclass_confusion_matrix_dense_default_batch_impl.i */
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
//  Declaration of template class that computes multi-class confusion matrix.
//--
*/

#ifndef __MULTICLASS_CONFUSION_MATRIX_DEFAULT_IMPL_I__
#define __MULTICLASS_CONFUSION_MATRIX_DEFAULT_IMPL_I__

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
namespace multiclass_confusion_matrix
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
void MultiClassConfusionMatrixKernel<method, algorithmFPType, cpu>::compute(const NumericTable *predictedLabelsTable,
                                                                            const NumericTable *groundTruthLabelsTable,
                                                                            NumericTable *confusionMatrixTable,
                                                                            NumericTable *accuracyMeasuresTable,
                                                                            const multiclass_confusion_matrix::Parameter *parameter)
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

    size_t nClasses = parameter->nClasses;
    algorithmFPType invNClasses = 1.0 / (algorithmFPType)nClasses;
    algorithmFPType beta = parameter->beta;
    algorithmFPType beta2 = beta * beta;

    /* Get memory to write the results */
    size_t nClassesReturned = mtConfusionMatrix.getBlockOfRows(0, nClasses, &confusionMatrixData);
    if (nClassesReturned < nClasses)
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

    /* Compute confusion matrix for multi-class classifier */
    for (size_t i = 0; i < nVectors; i++)
    {
        size_t predictedLabel   = (size_t)predictedLabelsData[i];
        size_t groundTruthLabel = (size_t)groundTruthLabelsData[i];

        confusionMatrixData[groundTruthLabel * nClasses + predictedLabel]++;
    }

    algorithmFPType *tp, *fp, *tn, *fn;
    tp = (algorithmFPType *)daal::services::daal_malloc(nClasses * sizeof(algorithmFPType));
    fp = (algorithmFPType *)daal::services::daal_malloc(nClasses * sizeof(algorithmFPType));
    tn = (algorithmFPType *)daal::services::daal_malloc(nClasses * sizeof(algorithmFPType));
    fn = (algorithmFPType *)daal::services::daal_malloc(nClasses * sizeof(algorithmFPType));
    if (!tp || !fp || !tn || !fn)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    for (size_t i = 0; i < nClasses; i++)
    {
        tp[i] = confusionMatrixData[i*nClasses + i];
        fp[i] = -tp[i];
        fn[i] = -tp[i];
        for (size_t j = 0; j < nClasses; j++)
        {
            fn[i] += confusionMatrixData[i*nClasses + j];
            fp[i] += confusionMatrixData[j*nClasses + i];
        }
        tn[i] = fpNVectors - tp[i] - fp[i] - fn[i];
    }

    service_memset<algorithmFPType, cpu>(accuracyMeasuresData, zero, 8);
    algorithmFPType tpSum = zero;
    for (size_t i = 0; i < nClasses; i++)
    {
        /* Average accuracy */
        accuracyMeasuresData[0] += (tp[i] + tn[i]);
        /* Error rate */
        accuracyMeasuresData[1] += (fp[i] + fn[i]);

        tpSum += tp[i];

        /* Micro Precision. Compute denominator */
        accuracyMeasuresData[2] += (tp[i] + fp[i]);
        /* Micro Recall. Compute denominator */
        accuracyMeasuresData[3] += (tp[i] + fn[i]);
        /* Macro Precision */
        accuracyMeasuresData[5] += (tp[i] / (tp[i] + fp[i]));
        /* Macro Recall */
        accuracyMeasuresData[6] += (tp[i] / (tp[i] + fn[i]));
    }
    /* Average accuracy */
    accuracyMeasuresData[0] *= (invNVectors * invNClasses);
    /* Error rate */
    accuracyMeasuresData[1] *= (invNVectors * invNClasses);
    /* Micro Precision */
    accuracyMeasuresData[2]  = tpSum / accuracyMeasuresData[2];
    /* Micro Recall */
    accuracyMeasuresData[3]  = tpSum / accuracyMeasuresData[3];
    /* Micro F-score */
    accuracyMeasuresData[4]  = ((beta2 + 1.0) * accuracyMeasuresData[2] * accuracyMeasuresData[3]) /
                               (beta2 * accuracyMeasuresData[2] + accuracyMeasuresData[3]);
    /* Macro Precision */
    accuracyMeasuresData[5] *= invNClasses;
    /* Macro Recall */
    accuracyMeasuresData[6] *= invNClasses;
    /* Macro F-score */
    accuracyMeasuresData[7]  = ((beta2 + 1.0) * accuracyMeasuresData[5] * accuracyMeasuresData[6]) /
                               (beta2 * accuracyMeasuresData[5] + accuracyMeasuresData[6]);

    daal::services::daal_free(tp);
    daal::services::daal_free(fp);
    daal::services::daal_free(tn);
    daal::services::daal_free(fn);

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
