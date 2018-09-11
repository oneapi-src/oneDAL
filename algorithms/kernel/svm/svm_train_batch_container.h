/* file: svm_train_batch_container.h */
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
//  Implementation of SVM training algorithm container.
//--
*/

#include "svm_train.h"
#include "svm_train_kernel.h"
#include "svm_train_boser_kernel.h"
#include "classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
/**
*  \brief Initialize list of SVM kernels with implementations for supported architectures
*/
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVMTrainImpl, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    svm::training::Result *result = static_cast<svm::training::Result *>(_res);
    size_t na = input->size();

    NumericTablePtr x = input->get(classifier::training::data);
    NumericTablePtr y = input->get(classifier::training::labels);

    daal::algorithms::Model *r = static_cast<daal::algorithms::Model *>(result->get(classifier::training::model).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::SVMTrainImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, x, *y, r, par);
}

} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
