/* file: decision_tree_classification_predict_dense_default_batch_container.h */
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
//  Implementation of Decision tree algorithm container - a class that contains fast Decision tree prediction kernels for supported
//  architectures.
//--
*/

#include "decision_tree_classification_predict.h"
#include "decision_tree_classification_predict_dense_default_batch.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace prediction
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::DecisionTreePredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const classifier::prediction::Input * const input = static_cast<const classifier::prediction::Input *>(_in);
    classifier::prediction::Result * const result = static_cast<classifier::prediction::Result *>(_res);

    const data_management::NumericTableConstPtr a = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m = input->get(classifier::prediction::model);
    const data_management::NumericTablePtr r = result->get(classifier::prediction::prediction);

    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DecisionTreePredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), \
                       compute, a.get(), m.get(), r.get(), par);
}

} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
