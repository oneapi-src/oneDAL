/* file: binary_confusion_matrix_dense_default_batch_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#ifndef __BINARY_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __BINARY_CONFUSION_MATRIX_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "binary_confusion_matrix_types.h"
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
namespace binary_confusion_matrix
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
class BinaryConfusionMatrixKernel : public Kernel
{
public:
    virtual ~BinaryConfusionMatrixKernel() {}

    services::Status compute(const NumericTable *predictedLabels, const NumericTable *groundTruthLabels,
                             NumericTable *confusionMatrix, NumericTable *accuracyMeasures,
                             const binary_confusion_matrix::Parameter *parameter);
};

}
}
}
}
}
}

#endif
