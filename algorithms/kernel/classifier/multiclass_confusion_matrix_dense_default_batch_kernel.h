/* file: multiclass_confusion_matrix_dense_default_batch_kernel.h */
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

template<Method method, typename algorithmFPType, CpuType cpu>
class MultiClassConfusionMatrixKernel : public Kernel
{
public:
    virtual ~MultiClassConfusionMatrixKernel() {}

    services::Status compute(const NumericTable *predictedLabels, const NumericTable *groundTruthLabels,
                             NumericTable *confusionMatrix, NumericTable *accuracyMeasures,
                             const multiclass_confusion_matrix::Parameter *parameter);
};

}
}
}
}
}
}

#endif
