/* file: multiclassclassifier_train_batch_container.h */
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

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassClassifierTrainKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *a[2];
    a[0] = static_cast<NumericTable *>(input->get(classifier::training::data).get());
    a[1] = static_cast<NumericTable *>(input->get(classifier::training::labels).get());
    multi_class_classifier::Model *r = static_cast<multi_class_classifier::Model *>(result->get(classifier::training::model).get());

    multi_class_classifier::Parameter *par = static_cast<multi_class_classifier::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassClassifierTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a[0], a[1], r,
                       par);
}

} // namespace training

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal
