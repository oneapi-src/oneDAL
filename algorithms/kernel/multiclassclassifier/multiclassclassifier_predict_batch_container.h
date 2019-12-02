/* file: multiclassclassifier_predict_batch_container.h */
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
//  Implementation of Multi-class classifier prediction algorithm container --
//  a class that contains Multi-class classifier kernels for supported
//  architectures.
//--
*/

#include "multi_class_classifier_predict.h"
#include "multiclassclassifier_predict_kernel.h"
#include "multiclassclassifier_predict_mccwu_kernel.h"
#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace interface2
{
template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassClassifierPredictKernel, pmethod, tmethod, algorithmFPType,
                              classifier::prediction::interface2::Batch, multi_class_classifier::interface2::Parameter);
}

template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
services::Status BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::compute()
{
    const classifier::prediction::Input * input = static_cast<const classifier::prediction::Input *>(_in);
    classifier::prediction::Result * result     = static_cast<classifier::prediction::Result *>(_res);

    const NumericTable * a                  = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    const multi_class_classifier::Model * m = static_cast<const multi_class_classifier::Model *>(input->get(classifier::prediction::model).get());
    NumericTable * r[1];
    r[0] = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());

    const daal::algorithms::Parameter * par = _par;
    daal::services::Environment::env & env  = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassClassifierPredictKernel,
                       __DAAL_KERNEL_ARGUMENTS(pmethod, tmethod, algorithmFPType, classifier::prediction::interface2::Batch,
                                               multi_class_classifier::interface2::Parameter),
                       compute, a, m, r[0], par);
}

} // namespace interface2

} // namespace prediction

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal
