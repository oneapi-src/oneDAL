/* file: binary_confusion_matrix_dense_default_batch_impl.i */
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
//  Declaration of template class that computes binary confusion matrix.
//--
*/

#ifndef __BINARY_CONFUSION_MATRIX_DEFAULT_IMPL_I__
#define __BINARY_CONFUSION_MATRIX_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_numeric_table.h"

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

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template<Method method, typename algorithmFPType, CpuType cpu>
services::Status BinaryConfusionMatrixKernel<method, algorithmFPType, cpu>::compute(const NumericTable *predictedLabelsTable,
                                                                                    const NumericTable *groundTruthLabelsTable,
                                                                                    NumericTable *confusionMatrixTable,
                                                                                    NumericTable *accuracyMeasuresTable,
                                                                                    const binary_confusion_matrix::Parameter *parameter)
{
    const algorithmFPType zero = 0.0;
    const size_t nVectors = predictedLabelsTable->getNumberOfRows();

    /* Get input data */
    ReadColumns<algorithmFPType, cpu> mtPredictedLabels(*const_cast<NumericTable*>(predictedLabelsTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtPredictedLabels);
    ReadColumns<algorithmFPType, cpu> mtGroundTruthLabels(*const_cast<NumericTable*>(groundTruthLabelsTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtGroundTruthLabels);

    /* Get memory to write the results */
    const size_t nClasses = 2;
    WriteOnlyRows<int, cpu> mtConfusionMatrix(confusionMatrixTable, 0, nClasses);
    DAAL_CHECK_BLOCK_STATUS(mtConfusionMatrix);
    WriteOnlyRows<algorithmFPType, cpu> mtAccuracyMeasures(accuracyMeasuresTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtAccuracyMeasures);

    const algorithmFPType *predictedLabelsData = mtPredictedLabels.get();
    const algorithmFPType *groundTruthLabelsData = mtGroundTruthLabels.get();
    int *confusionMatrixData = mtConfusionMatrix.get();
    algorithmFPType *accuracyMeasuresData = mtAccuracyMeasures.get();

    algorithmFPType beta = parameter->beta;
    algorithmFPType beta2 = beta * beta;
    service_memset<int, cpu>(confusionMatrixData, 0, nClasses * nClasses);

    /* Compute confusion matrix for two-class classifier */

    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nVectors; i++)
    {
        const int predictedLabel   = ((predictedLabelsData[i]   > zero) ? 0 : 1);
        const int groundTruthLabel = ((groundTruthLabelsData[i] > zero) ? 0 : 1);
        ++confusionMatrixData[groundTruthLabel * 2 + predictedLabel];
    }

    const algorithmFPType tp(confusionMatrixData[0]);
    const algorithmFPType fn(confusionMatrixData[1]);
    const algorithmFPType fp(confusionMatrixData[2]);
    const algorithmFPType tn(confusionMatrixData[3]);

    const algorithmFPType invNVectors = 1.0 / algorithmFPType(nVectors);
    /* Accuracy */
    accuracyMeasuresData[0] = (tp + tn) * invNVectors;
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
    return Status();
}

}
}
}
}
}
}

#endif
