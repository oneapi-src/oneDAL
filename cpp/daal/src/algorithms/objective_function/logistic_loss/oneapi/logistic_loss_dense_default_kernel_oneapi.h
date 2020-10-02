/* file: logistic_loss_dense_default_kernel_oneapi.h */
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
//  Implementation of Logistic Loss Batch Kernel for GPU.
//--
*/

#ifndef __LOGISTIC_LOSS_DENSE_DEFAULT_KERNEL_ONEAPI_H__
#define __LOGISTIC_LOSS_DENSE_DEFAULT_KERNEL_ONEAPI_H__

#include "src/sycl/blas_gpu.h"
#include "src/algorithms/objective_function/common/oneapi/objective_function_utils_oneapi.h"
#include "algorithms/optimization_solver/objective_function/logistic_loss_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{
template <typename algorithmFPType, Method method>
class LogLossKernelOneAPI : public Kernel
{};

template <typename algorithmFPType>
class LogLossKernelOneAPI<algorithmFPType, defaultDense> : public Kernel
{
    using HelperObjectiveFunction = objective_function::internal::HelperObjectiveFunction<algorithmFPType>;

public:
    services::Status compute(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value,
                             NumericTable * hessian, NumericTable * gradient, NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
                             NumericTable * lipschitzConstant, Parameter * parameter);

    static services::Status applyBeta(const services::internal::Buffer<algorithmFPType> & x, const services::internal::Buffer<algorithmFPType> & beta,
                                      services::internal::Buffer<algorithmFPType> & xb, const uint32_t n, const uint32_t p, const uint32_t offset);

    static services::Status applyGradient(const services::internal::Buffer<algorithmFPType> & x,
                                          const services::internal::Buffer<algorithmFPType> & sub,
                                          services::internal::Buffer<algorithmFPType> & gradient, const algorithmFPType alpha, const uint32_t n,
                                          const uint32_t p, const algorithmFPType beta, const uint32_t offset);

    static services::Status applyHessian(const services::internal::Buffer<algorithmFPType> & x,
                                         const services::internal::Buffer<algorithmFPType> & sigma, const uint32_t n, const uint32_t p,
                                         services::internal::Buffer<algorithmFPType> & h, const uint32_t nBeta, const uint32_t offset,
                                         const algorithmFPType alpha);

    // TODO: move in common services
    static services::Status logLoss(const services::internal::Buffer<algorithmFPType> & y, const services::internal::Buffer<algorithmFPType> & sigma,
                                    services::internal::Buffer<algorithmFPType> & result, const uint32_t n);

    // TODO: move in common services
    static services::Status sigmoids(const services::internal::Buffer<algorithmFPType> & x, services::internal::Buffer<algorithmFPType> & result,
                                     const uint32_t n, bool calculateInverse = false);

    static services::Status betaIntercept(const services::internal::Buffer<algorithmFPType> & arg, services::internal::Buffer<algorithmFPType> & x,
                                          const uint32_t n);

private:
    services::Status doCompute(const uint32_t nBatch, const uint32_t nFeatures, const daal::services::internal::Buffer<algorithmFPType> & xBuff,
                               const daal::services::internal::Buffer<algorithmFPType> & yBuff,
                               const daal::services::internal::Buffer<algorithmFPType> & argBuff, NumericTable * valueNT, NumericTable * gradientNT,
                               NumericTable * hessianNT, NumericTable * nonSmoothTermValueNT, NumericTable * proximalProjectionNT,
                               NumericTable * lipschitzConstantNT, const algorithmFPType l1reg, const algorithmFPType l2reg, const bool interceptFlag,
                               const bool isSourceData);

    static services::Status hessianRegulization(services::internal::Buffer<algorithmFPType> & h, const uint32_t nBeta, const algorithmFPType l2);

    static services::Status hessianIntercept(const services::internal::Buffer<algorithmFPType> & x,
                                             const services::internal::Buffer<algorithmFPType> & sigma, const uint32_t n, const uint32_t p,
                                             services::internal::Buffer<algorithmFPType> & h, const uint32_t nBeta, const algorithmFPType alpha);

    static void buildProgram(services::internal::sycl::ClKernelFactoryIface & factory);

    services::internal::sycl::UniversalBuffer _uX;
    services::internal::sycl::UniversalBuffer _uY;
    services::internal::sycl::UniversalBuffer _fUniversal;
    services::internal::sycl::UniversalBuffer _sigmoidUniversal;
    services::internal::sycl::UniversalBuffer _subSigmoidYUniversal;
};

} // namespace internal
} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
