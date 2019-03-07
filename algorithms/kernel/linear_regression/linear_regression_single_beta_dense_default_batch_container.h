/* file: linear_regression_single_beta_dense_default_batch_container.h */
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
//  Implementation of the container for the multi-class confusion matrix.
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/linear_regression/linear_regression_single_beta_batch.h"
#include "linear_regression_single_beta_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{

using namespace daal::data_management;

namespace internal
{

const NumericTable* getXtXTable(const linear_regression::Model& model, bool& bModelNe);

}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SingleBetaKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input* input = static_cast<Input* >(_in);
    Result* result = static_cast<Result* >(_res);
    Parameter* par = static_cast<Parameter* >(_par);

    data_management::DataCollection* coll = result->get(betaCovariances).get();
    internal::SingleBetaOutput out(coll->size());
    out.rms = result->get(rms).get();
    out.variance = result->get(variance).get();
    out.zScore = result->get(zScore).get();
    out.confidenceIntervals = result->get(confidenceIntervals).get();
    out.inverseOfXtX = result->get(inverseOfXtX).get();
    for(size_t i = 0; i < coll->size(); ++i)
        out.betaCovariances[i] = dynamic_cast<NumericTable*>((*coll)[i].get());

    const auto pModel = input->get(model).get();
    bool bModelNe = false;
    const auto pXtx = internal::getXtXTable(*pModel, bModelNe);
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::SingleBetaKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
        compute, input->get(expectedResponses).get(), input->get(predictedResponses).get(), pModel->getNumberOfFeatures(),
        pModel->getBeta().get(), pXtx, bModelNe, par->accuracyThreshold, par->alpha, out);
}

}
}
}
}
}

#endif
