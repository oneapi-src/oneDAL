/* file: svm_predict_batch_container.h */
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
//  Implementation of SVM prediction algorithm container.
//--
*/

#include "svm_predict.h"
#include "svm_predict_kernel.h"
#include "classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
/**
*  \brief Initialize list of SVM kernels with implementations for supported architectures
*/
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVMPredictImpl, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Input *input = static_cast<classifier::prediction::Input *>(_in);
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);

    NumericTablePtr a = input->get(classifier::prediction::data);
    daal::algorithms::Model *m = static_cast<daal::algorithms::Model *>(input->get(classifier::prediction::model).get());
    NumericTablePtr r = result->get(classifier::prediction::prediction);

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVMPredictImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, *r, par);
}

} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
