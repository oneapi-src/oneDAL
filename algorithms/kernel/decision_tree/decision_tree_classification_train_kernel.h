/* file: decision_tree_classification_train_kernel.h */
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
//  Declaration of structure containing kernels for K-Nearest Neighbors training.
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_TRAIN_KERNEL_H__
#define __DECISION_TREE_CLASSIFICATION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "decision_tree_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::services;

template <typename algorithmFPType, training::Method method, CpuType cpu>
class DecisionTreeTrainBatchKernel
{};

template <typename algorithmFPType, CpuType cpu>
class DecisionTreeTrainBatchKernel<algorithmFPType, training::defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * x, const NumericTable * y, const NumericTable * px, const NumericTable * py,
                 decision_tree::classification::Model * r, const daal::algorithms::Parameter * par);
};

} // namespace internal
} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
