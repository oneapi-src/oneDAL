/* file: multiclassclassifier_train_batch_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of Multi-class classifier algorithm container -- a class that contains
//  Multi-class classifier kernels for supported architectures.
//--
*/

#include "multi_class_classifier_train.h"
#include "multiclassclassifier_train_kernel.h"
#include "multiclassclassifier_train_oneagainstone_kernel.h"
#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{

template<typename AlgorithmFPType, Method method, CpuType cpu>
BatchContainer<AlgorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassClassifierTrainKernel, method, AlgorithmFPType);
}

template<typename AlgorithmFPType, Method method, CpuType cpu>
BatchContainer<AlgorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename AlgorithmFPType, Method method, CpuType cpu>
void BatchContainer<AlgorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *a[2];
    a[0] = static_cast<NumericTable *>(input->get(classifier::training::data).get());;
    a[1] = static_cast<NumericTable *>(input->get(classifier::training::labels).get());;
    multi_class_classifier::Model *r = static_cast<multi_class_classifier::Model *>(result->get(classifier::training::model).get());

    multi_class_classifier::Parameter *par = static_cast<multi_class_classifier::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassClassifierTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, AlgorithmFPType), compute, a[0], a[1], r,
                       par);
}

} // namespace training

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal
