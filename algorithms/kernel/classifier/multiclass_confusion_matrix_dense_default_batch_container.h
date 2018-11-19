/* file: multiclass_confusion_matrix_dense_default_batch_container.h */
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
//  Implementation of the container for the multi-class confusion matrix.
//--
*/

#ifndef __MULTICLASS_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __MULTICLASS_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/classifier/multiclass_confusion_matrix_batch.h"
#include "multiclass_confusion_matrix_dense_default_batch_kernel.h"

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
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassConfusionMatrixKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);
    NumericTable *predictedLabelsTable   = static_cast<NumericTable *>(input->get(predictedLabels  ).get());
    NumericTable *groundTruthLabelsTable = static_cast<NumericTable *>(input->get(groundTruthLabels).get());

    NumericTable *confusionMatrixTable  = static_cast<NumericTable *>(result->get(confusionMatrix).get());
    NumericTable *accuracyMeasuresTable = static_cast<NumericTable *>(result->get(multiClassMetrics).get());

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassConfusionMatrixKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),   \
                       compute, predictedLabelsTable, groundTruthLabelsTable, confusionMatrixTable, accuracyMeasuresTable, parameter);
}

}
}
}
}
}

#endif
