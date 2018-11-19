/* file: decision_tree_classification_train_container.h */
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
//  Implementation of K-Nearest Neighbors container.
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __DECISION_TREE_CLASSIFICATION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_shared_ptr.h"
#include "decision_tree_classification_training_batch.h"
#include "decision_tree_classification_train_kernel.h"
#include "decision_tree_classification_model_impl.h"

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

using namespace daal::data_management;

/**
 *  \brief Initialize list of Decision tree kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DecisionTreeTrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate Decision tree model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const decision_tree::classification::training::Input * const input = static_cast<decision_tree::classification::training::Input *>(_in);
    Result * const result = static_cast<Result *>(_res);

    const NumericTableConstPtr x = input->get(classifier::training::data);
    const NumericTableConstPtr y = input->get(classifier::training::labels);
    const NumericTableConstPtr px = input->get(dataForPruning);
    const NumericTableConstPtr py = input->get(labelsForPruning);

    const ModelPtr r = result->get(classifier::training::model);

    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DecisionTreeTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),    \
                       compute, x.get(), y.get(), px.get(), py.get(), r.get(), par);
}

} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
