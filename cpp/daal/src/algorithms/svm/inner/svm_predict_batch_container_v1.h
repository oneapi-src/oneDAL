/* file: svm_predict_batch_container_v1.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of SVM prediction algorithm container.
//--
*/

#include "algorithms/svm/svm_predict.h"
#include "src/algorithms/svm/svm_predict_kernel.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace interface1
{
/**
*  \brief Initialize list of SVM kernels with implementations for supported architectures
*/
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVMPredictImpl, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Input * input   = static_cast<classifier::prediction::Input *>(_in);
    classifier::prediction::Result * result = static_cast<classifier::prediction::Result *>(_res);

    data_management::NumericTablePtr a = input->get(classifier::prediction::data);
    Model * m                          = static_cast<Model *>(input->get(classifier::prediction::model).get());
    data_management::NumericTablePtr r = result->get(classifier::prediction::prediction);

    services::Environment::env & env = *_env;
    svm::interface1::Parameter * par = static_cast<svm::interface1::Parameter *>(_par);
    svm::interface2::Parameter par2;

    par2.C                 = par->C;
    par2.accuracyThreshold = par->accuracyThreshold;
    par2.tau               = par->tau;
    par2.maxIterations     = par->maxIterations;
    par2.cacheSize         = par->cacheSize;
    par2.doShrinking       = par->doShrinking;
    par2.shrinkingStep     = par->shrinkingStep;
    par2.kernel            = par->kernel;

    __DAAL_CALL_KERNEL(env, internal::SVMPredictImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, *r, &par2);
}
} // namespace interface1
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
