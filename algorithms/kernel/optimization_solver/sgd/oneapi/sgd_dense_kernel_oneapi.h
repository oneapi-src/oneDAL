/* file: sgd_dense_kernel_oneapi.h */
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

//++
//  Declaration of template function that calculate sgd.
//--

#ifndef __SGD_DENSE_KERNEL_ONEAPI_H__
#define __SGD_DENSE_KERNEL_ONEAPI_H__

#include "algorithms/optimization_solver/sgd/sgd_batch.h"
#include "algorithms/kernel/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel/optimization_solver/iterative_solver_kernel.h"
#include "algorithms/kernel/optimization_solver/sgd/sgd_dense_kernel.h"
#include "service/kernel/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
using namespace daal::data_management;

template <typename algorithmFPType, Method method>
class SGDKernelOneAPI : public Kernel
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTablePtr minimum, NumericTable * nIterations,
                             Parameter<method> * parameter, NumericTable * learningRateSequence, NumericTable * batchIndices,
                             OptionalArgument * optionalArgument, OptionalArgument * optionalResult, engines::BatchBase & engine)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
class SGDKernelOneAPI<algorithmFPType, miniBatch> : public Kernel
{
public:
    services::Status compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTablePtr minimum, NumericTable * nIterations,
                             Parameter<miniBatch> * parameter, NumericTable * learningRateSequence, NumericTable * batchIndices,
                             OptionalArgument * optionalArgument, OptionalArgument * optionalResult, engines::BatchBase & engine);

private:
    static services::Status makeStep(const uint32_t argumentSize, const services::Buffer<algorithmFPType> & prevWorkValueBuff,
                                     const services::Buffer<algorithmFPType> & gradientBuff, services::Buffer<algorithmFPType> & workValueBuff,
                                     const algorithmFPType learningRate, const algorithmFPType consCoeff);

    static services::Status vectorNorm(const services::Buffer<algorithmFPType> & x, const uint32_t n, algorithmFPType & norm);

    static void buildProgram(oneapi::internal::ClKernelFactoryIface & factory);

    enum IndicesStatus
    {
        random = 0, /*!< Indices of the terms are generated randomly */
        user   = 1, /*!< Indices of the terms are provided by user */
        all    = 2  /*!< All objective function terms are used for computations */
    };
};

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
