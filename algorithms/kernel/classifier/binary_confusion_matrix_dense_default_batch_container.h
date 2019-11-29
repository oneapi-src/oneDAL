/* file: binary_confusion_matrix_dense_default_batch_container.h */
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
//  Implementation of the container for the binary confusion matrix.
//--
*/

#ifndef __BINARY_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __BINARY_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/classifier/binary_confusion_matrix_batch.h"
#include "binary_confusion_matrix_dense_default_batch_kernel.h"

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
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BinaryConfusionMatrixKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input                         = static_cast<Input *>(_in);
    Result * result                       = static_cast<Result *>(_res);
    Parameter * parameter                 = static_cast<Parameter *>(_par);
    NumericTable * predictedLabelsTable   = static_cast<NumericTable *>(input->get(predictedLabels).get());
    NumericTable * groundTruthLabelsTable = static_cast<NumericTable *>(input->get(groundTruthLabels).get());

    NumericTable * confusionMatrixTable  = static_cast<NumericTable *>(result->get(confusionMatrix).get());
    NumericTable * accuracyMeasuresTable = static_cast<NumericTable *>(result->get(binaryMetrics).get());

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::BinaryConfusionMatrixKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, predictedLabelsTable,
                       groundTruthLabelsTable, confusionMatrixTable, accuracyMeasuresTable, parameter);
}

} // namespace binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
