/* file: multiclass_confusion_matrix_dense_default_batch_kernel.h */
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

#ifndef __MULTICLASS_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __MULTICLASS_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "multiclass_confusion_matrix_types.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

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
class MultiClassConfusionMatrixKernel : public Kernel
{
public:
    virtual ~MultiClassConfusionMatrixKernel() {}

    services::Status compute(const NumericTable * predictedLabels, const NumericTable * groundTruthLabels, NumericTable * confusionMatrix,
                             NumericTable * accuracyMeasures, const multiclass_confusion_matrix::Parameter * parameter);
};

} // namespace internal
} // namespace multiclass_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
