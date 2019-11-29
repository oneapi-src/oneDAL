/* file: multiclass_confusion_matrix_dense_default_batch_impl.i */
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
//  Declaration of template class that computes multi-class confusion matrix.
//--
*/

#ifndef __MULTICLASS_CONFUSION_MATRIX_DEFAULT_IMPL_I__
#define __MULTICLASS_CONFUSION_MATRIX_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_numeric_table.h"

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
template <Method method, typename algorithmFPType, CpuType cpu>
Status MultiClassConfusionMatrixKernel<method, algorithmFPType, cpu>::compute(const NumericTable * predictedLabelsTable,
                                                                              const NumericTable * groundTruthLabelsTable,
                                                                              NumericTable * confusionMatrixTable,
                                                                              NumericTable * accuracyMeasuresTable,
                                                                              const multiclass_confusion_matrix::Parameter * parameter)
{
    const algorithmFPType zero = 0.0;

    const size_t nVectors = predictedLabelsTable->getNumberOfRows();

    /* Get input data */
    ReadColumns<algorithmFPType, cpu> mtPredictedLabels(*const_cast<NumericTable *>(predictedLabelsTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtPredictedLabels);
    ReadColumns<algorithmFPType, cpu> mtGroundTruthLabels(*const_cast<NumericTable *>(groundTruthLabelsTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtGroundTruthLabels);

    /* Get memory to write the results */
    const size_t nClasses = parameter->nClasses;
    WriteOnlyRows<int, cpu> mtConfusionMatrix(confusionMatrixTable, 0, nClasses);
    DAAL_CHECK_BLOCK_STATUS(mtConfusionMatrix);
    WriteOnlyRows<algorithmFPType, cpu> mtAccuracyMeasures(accuracyMeasuresTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtAccuracyMeasures);

    const algorithmFPType * predictedLabelsData   = mtPredictedLabels.get();
    const algorithmFPType * groundTruthLabelsData = mtGroundTruthLabels.get();
    int * confusionMatrixData                     = mtConfusionMatrix.get();
    algorithmFPType * accuracyMeasuresData        = mtAccuracyMeasures.get();

    const algorithmFPType invNVectors = 1.0 / algorithmFPType(nVectors);
    const algorithmFPType invNClasses = 1.0 / algorithmFPType(nClasses);
    algorithmFPType beta              = parameter->beta;
    algorithmFPType beta2             = beta * beta;

    service_memset<int, cpu>(confusionMatrixData, 0, nClasses * nClasses);

    /* Compute confusion matrix for multi-class classifier */
    for (size_t i = 0; i < nVectors; i++)
    {
        DAAL_CHECK(predictedLabelsData[i] >= 0 && predictedLabelsData[i] < nClasses, ErrorIncorrectClassLabels)
        DAAL_CHECK(groundTruthLabelsData[i] >= 0 && groundTruthLabelsData[i] < nClasses, ErrorIncorrectClassLabels)

        size_t predictedLabel   = (size_t)predictedLabelsData[i];
        size_t groundTruthLabel = (size_t)groundTruthLabelsData[i];

        confusionMatrixData[groundTruthLabel * nClasses + predictedLabel]++;
    }

    TArray<algorithmFPType, cpu> aTp(nClasses);
    TArray<algorithmFPType, cpu> aFp(nClasses);
    TArray<algorithmFPType, cpu> aTn(nClasses);
    TArray<algorithmFPType, cpu> aFn(nClasses);

    algorithmFPType * tp = aTp.get();
    algorithmFPType * fp = aFp.get();
    algorithmFPType * tn = aTn.get();
    algorithmFPType * fn = aFn.get();
    DAAL_CHECK(tp && fp && tn && fn, ErrorMemoryAllocationFailed);

    algorithmFPType fpNVectors(nVectors);
    for (size_t i = 0; i < nClasses; i++)
    {
        tp[i] = confusionMatrixData[i * nClasses + i];
        fp[i] = -tp[i];
        fn[i] = -tp[i];
        for (size_t j = 0; j < nClasses; j++)
        {
            fn[i] += confusionMatrixData[i * nClasses + j];
            fp[i] += confusionMatrixData[j * nClasses + i];
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
    accuracyMeasuresData[2] = tpSum / accuracyMeasuresData[2];
    /* Micro Recall */
    accuracyMeasuresData[3] = tpSum / accuracyMeasuresData[3];
    /* Micro F-score */
    accuracyMeasuresData[4] =
        ((beta2 + 1.0) * accuracyMeasuresData[2] * accuracyMeasuresData[3]) / (beta2 * accuracyMeasuresData[2] + accuracyMeasuresData[3]);
    /* Macro Precision */
    accuracyMeasuresData[5] *= invNClasses;
    /* Macro Recall */
    accuracyMeasuresData[6] *= invNClasses;
    /* Macro F-score */
    accuracyMeasuresData[7] =
        ((beta2 + 1.0) * accuracyMeasuresData[5] * accuracyMeasuresData[6]) / (beta2 * accuracyMeasuresData[5] + accuracyMeasuresData[6]);
    return Status();
}

} // namespace internal
} // namespace multiclass_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
