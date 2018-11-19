/* file: gbt_classification_train_container.h */
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
//  Implementation of gradient boosted trees container.
//--
*/

#ifndef __GBT_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __GBT_CLASSIFICATION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "gbt_classification_training_types.h"
#include "gbt_classification_training_batch.h"
#include "gbt_classification_train_kernel.h"
#include "gbt_classification_model_impl.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ClassificationTrainBatchKernel, algorithmFPType, method);
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
    Result *result = static_cast<Result *>(_res);

    NumericTable *x = input->get(classifier::training::data).get();
    NumericTable *y = input->get(classifier::training::labels).get();

    gbt::classification::Model *m = result->get(classifier::training::model).get();

    const gbt::classification::training::Parameter *par =
        static_cast<gbt::classification::training::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;
    daal::algorithms::engines::internal::BatchBaseImpl* engine = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(par->engine.get());

    __DAAL_CALL_KERNEL(env, internal::ClassificationTrainBatchKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, daal::services::internal::hostApp(*input), x, y, *m, *result, *par, *engine);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result *result = static_cast<Result *>(_res);
    gbt::classification::Model *m = result->get(classifier::training::model).get();
    gbt::classification::internal::ModelImpl* pImpl = dynamic_cast<gbt::classification::internal::ModelImpl*>(m);
    DAAL_ASSERT(pImpl);
    pImpl->clear();
    return services::Status();
}

}
}
}
}
}
#endif
