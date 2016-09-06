/* file: linear_regression_predict_dense_default_batch_container.h */
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
//  Implementation of linear regression algorithm container -- a class
//  that contains fast linear regression prediction kernels
//  for supported architectures.
//--
*/

#include "linear_regression_predict.h"
#include "linear_regression_predict_dense_default_batch.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace prediction
{

template <typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::LinearRegressionPredictKernel, interm, method);
}

template <typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename interm, Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(data).get());
    daal::algorithms::Model *m = static_cast<daal::algorithms::Model *>(input->get(model).get());
    NumericTable *r = static_cast<NumericTable *>(result->get(prediction).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LinearRegressionPredictKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, a, m, r, par);
}

}
}
}
} // namespace daal
