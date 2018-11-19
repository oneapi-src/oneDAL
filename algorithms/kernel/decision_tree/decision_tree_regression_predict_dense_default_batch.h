/* file: decision_tree_regression_predict_dense_default_batch.h */
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
//  Declaration of template function that computes Decision tree prediction results.
//--
*/

#ifndef __DECISION_TREE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __DECISION_TREE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "decision_tree_regression_predict.h"
#include "decision_tree_regression_model_impl.h"
#include "kernel.h"
#include "numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace prediction
{
namespace internal
{

using namespace daal::data_management;

template <typename algorithmFPType, prediction::Method method, CpuType cpu>
class DecisionTreePredictKernel : public daal::algorithms::Kernel
{};

template <typename algorithmFPType, CpuType cpu>
class DecisionTreePredictKernel<algorithmFPType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * x, const daal::algorithms::Model * m, NumericTable * y, const daal::algorithms::Parameter * par);
};

} // namespace internal
} // namespace prediction
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
