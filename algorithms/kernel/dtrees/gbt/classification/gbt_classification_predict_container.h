/* file: gbt_classification_predict_container.h */
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
//  Implementation of gradient boosted trees algorithm container -- a class
//  that contains fast gradient boosted trees prediction kernels
//  for supported architectures.
//--
*/

#include "gbt_classification_predict.h"
#include "gbt_classification_predict_kernel.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    gbt::classification::Model *m = static_cast<gbt::classification::Model *>(input->get(classifier::prediction::model).get());
    NumericTable *r = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());

    daal::services::Environment::env &env = *_env;
    const gbt::classification::prediction::Parameter *par = static_cast<gbt::classification::prediction::Parameter*>(_par);

    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, daal::services::internal::hostApp(*input),
        a, m, r, par->nClasses, par->nIterations);
}

}
}
}
}
} // namespace daal
