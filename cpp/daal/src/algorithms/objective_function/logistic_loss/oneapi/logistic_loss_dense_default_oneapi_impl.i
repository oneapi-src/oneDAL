/* file: logistic_loss_dense_default_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of Logistic Loss algorithm for GPU.
//--
*/

#include "src/algorithms/objective_function/logistic_loss/oneapi/cl_kernel/logistic_loss_dense_default.cl"
#include "src/services/service_utils.h"
#include "src/externals/service_math.h"

#include "src/externals/service_profiler.h"

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
using namespace daal::services::internal;
using namespace daal::services::internal::sycl;
using namespace daal::internal;

// Calculate X^T*beta
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::applyBeta(const services::internal::Buffer<algorithmFPType> & x,
                                                                               const services::internal::Buffer<algorithmFPType> & beta,
                                                                               services::internal::Buffer<algorithmFPType> & xb, const uint32_t n,
                                                                               const uint32_t p, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyBeta);
    DAAL_ASSERT(x.size() == size_t(n) * size_t(p)); // overflows checked in the algorithm
    DAAL_ASSERT(beta.size() >= size_t(offset) + size_t(p));
    DAAL_ASSERT(xb.size() >= size_t(n));
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans, n, 1, p, algorithmFPType(1), x,
                                           p, 0, beta, 1, offset, algorithmFPType(0), xb, 1, 0);
}

// Calculate X^T*(y - sigma) + 2*L2*beta
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::applyGradient(const services::internal::Buffer<algorithmFPType> & x,
                                                                                   const services::internal::Buffer<algorithmFPType> & sub,
                                                                                   services::internal::Buffer<algorithmFPType> & gradient,
                                                                                   const algorithmFPType alpha, const uint32_t n, const uint32_t p,
                                                                                   const algorithmFPType beta, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyGradient);
    DAAL_ASSERT(x.size() == size_t(n) * size_t(p)); // overflows checked in the algorithm
    DAAL_ASSERT(sub.size() >= size_t(n));
    DAAL_ASSERT(gradient.size() >= size_t(offset) + size_t(p));
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, p, 1, n, alpha, x, p, 0, sub, 1,
                                           0, beta, gradient, 1, offset);
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::applyHessian(
    const services::internal::Buffer<algorithmFPType> & x, const services::internal::Buffer<algorithmFPType> & sigma, const uint32_t n,
    const uint32_t p, services::internal::Buffer<algorithmFPType> & h, const uint32_t nBeta, const uint32_t offset, const algorithmFPType alpha)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyHessian);
    services::Status status;

    DAAL_ASSERT(x.size() == n * p); //overflows checked in the algorithm
    DAAL_ASSERT(h.size() == nBeta * nBeta);
    DAAL_ASSERT(sigma.size() == n);

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "hessian";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(8, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, x, AccessModeIds::read);
    args.set(1, p);
    args.set(2, sigma, AccessModeIds::read);
    args.set(3, n);
    args.set(4, h, AccessModeIds::write);
    args.set(5, nBeta);
    args.set(6, offset);
    args.set(7, alpha);

    KernelRange range(p, p);

    ctx.run(range, kernel, args, status);

    return services::Status();
}

// ylog(sigm) + (1-y)log(1-sigma)
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::logLoss(const services::internal::Buffer<algorithmFPType> & y,
                                                                             const services::internal::Buffer<algorithmFPType> & sigma,
                                                                             services::internal::Buffer<algorithmFPType> & result, const uint32_t n)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(logLoss);
    services::Status status;

    DAAL_ASSERT(y.size() == n);
    DAAL_ASSERT(sigma.size() == n);
    DAAL_ASSERT(result.size() == n);

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "logLoss";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(3, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, y, AccessModeIds::read);
    args.set(1, sigma, AccessModeIds::read);
    args.set(2, result, AccessModeIds::write);

    KernelRange range(n);
    ctx.run(range, kernel, args, status);
    return status;
}

// sigmoid(x) = 1/(1+exp(-x))
// if calculateInverse = true, x[i][1] = 1 - sigmoid(x[i][0])
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::sigmoids(const services::internal::Buffer<algorithmFPType> & x,
                                                                              services::internal::Buffer<algorithmFPType> & result, const uint32_t n,
                                                                              bool calculateInverse)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(sigmoids);
    services::Status status;

    DAAL_ASSERT(x.size() >= n);
    DAAL_ASSERT(calculateInverse ? result.size() >= 2 * n : result.size() >= n);

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "sigmoid";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    const algorithmFPType expThreshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, x, AccessModeIds::read);
    args.set(1, expThreshold);
    args.set(2, uint32_t(calculateInverse));
    args.set(3, result, AccessModeIds::write);

    KernelRange range(n);
    ctx.run(range, kernel, args, status);
    return status;
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::betaIntercept(const services::internal::Buffer<algorithmFPType> & arg,
                                                                                   services::internal::Buffer<algorithmFPType> & x, const uint32_t n)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(betaIntercept);
    services::Status status;

    // xb += b0
    const algorithmFPType zero = algorithmFPType(0);
    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setElem(0, zero, x));
    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::addVectorScalar(x, arg, 0, n));

    return status;
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::hessianIntercept(const services::internal::Buffer<algorithmFPType> & x,
                                                                                      const services::internal::Buffer<algorithmFPType> & sigma,
                                                                                      const uint32_t n, const uint32_t p,
                                                                                      services::internal::Buffer<algorithmFPType> & h,
                                                                                      const uint32_t nBeta, const algorithmFPType alpha)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(hessianIntercept);
    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    DAAL_ASSERT(x.size() == n * p); //overflows checked in the algorithm
    DAAL_ASSERT(h.size() == nBeta * nBeta);
    DAAL_ASSERT(sigma.size() == n);

    {
        const char * const kernelName = "hessianIntercept";
        KernelPtr kernel              = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(7, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, x, AccessModeIds::read);
        args.set(1, p);
        args.set(2, sigma, AccessModeIds::read);
        args.set(3, n);
        args.set(4, h, AccessModeIds::write);
        args.set(5, nBeta);
        args.set(6, alpha);

        KernelRange range(nBeta);

        ctx.run(range, kernel, args, status);
    }
    {
        // h[0][0] = alpha*sigma[i]*(1-sima[i])
        algorithmFPType h00           = algorithmFPType(0);
        const char * const kernelName = "hessianInterceptH0";

        KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelNDRange range(1);

        size_t workItemsPerGroup = 256;

        const size_t nWorkGroups = HelperObjectiveFunction::getWorkgroupsCount(n, workItemsPerGroup);

        KernelRange localRange(workItemsPerGroup);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nWorkGroups, workItemsPerGroup);
        KernelRange globalRange(nWorkGroups * workItemsPerGroup);

        range.local(localRange, status);
        range.global(globalRange, status);
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer buffer = ctx.allocate(idType, nWorkGroups, status);
        DAAL_CHECK_STATUS_VAR(status);
        services::internal::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, sigma, AccessModeIds::read);
        args.set(1, n);
        args.set(2, reductionBuffer, AccessModeIds::write);

        ctx.run(range, kernel, args, status);

        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::sumReduction(reductionBuffer, nWorkGroups, h00));

        h00 *= alpha;
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setElem(0, h00, h));
    }
    return services::Status();
}

// ylog(sigm) + (1-y)log(1-sigma)
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::hessianRegulization(services::internal::Buffer<algorithmFPType> & h,
                                                                                         const uint32_t nBeta, const algorithmFPType l2)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(hessianRegulization);

    DAAL_ASSERT(h.size() == nBeta * nBeta); //overflows checked in the algorithm

    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "hessianRegulization";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    const algorithmFPType beta = l2 * algorithmFPType(2);

    KernelArguments args(3, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, h, AccessModeIds::write);
    args.set(1, nBeta);
    args.set(2, beta);

    KernelRange range(nBeta - 1);

    ctx.run(range, kernel, args, status);

    return status;
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);

    services::Status status;
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_optimization_solver_logistic_loss_");
    cachekey.add(options);

    options.add(" -D LOCAL_SUM_SIZE=256 ");

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelLogLoss, options.c_str(), status);
    return status;
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::doCompute(
    const uint32_t nBatch, const uint32_t nFeatures, const daal::services::internal::Buffer<algorithmFPType> & xBuff,
    const daal::services::internal::Buffer<algorithmFPType> & yBuff, const daal::services::internal::Buffer<algorithmFPType> & argBuff,
    NumericTable * valueNT, NumericTable * gradientNT, NumericTable * hessianNT, NumericTable * nonSmoothTermValueNT,
    NumericTable * proximalProjectionNT, NumericTable * lipschitzConstantNT, const algorithmFPType l1reg, const algorithmFPType l2reg,
    const bool interceptFlag, const bool isSourceData)
{
    services::Status status;

    ExecutionContextIface & ctx = services::internal::getDefaultContext();

    const uint32_t nBeta = nFeatures + 1;
    DAAL_ASSERT(nBeta > nFeatures);
    const uint32_t ldX     = isSourceData ? nFeatures : nBeta;
    const uint32_t offsetX = isSourceData ? 1 : 0;

    const uint32_t n = nBatch;

    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    if (valueNT == nullptr && gradientNT == nullptr && hessianNT == nullptr)
    {
        return services::ErrorMethodNotImplemented;
    }

    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_fUniversal, n));
    services::internal::Buffer<algorithmFPType> fBuf = _fUniversal.get<algorithmFPType>();

    //f = X*b + b0
    DAAL_CHECK_STATUS(status, applyBeta(xBuff, argBuff, fBuf, n, ldX, offsetX));

    if (interceptFlag)
    {
        DAAL_CHECK_STATUS(status, betaIntercept(argBuff, fBuf, n));
    }

    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_sigmoidUniversal, n));
    services::internal::Buffer<algorithmFPType> sigmoidBuf = _sigmoidUniversal.get<algorithmFPType>();

    //s = exp(-f)
    DAAL_CHECK_STATUS(status, sigmoids(fBuf, sigmoidBuf, n));
    const algorithmFPType div = algorithmFPType(1) / algorithmFPType(n);

    if (valueNT)
    {
        DAAL_ASSERT(valueNT->getNumberOfRows() == 1);

        BlockDescriptor<algorithmFPType> vr;
        DAAL_CHECK_STATUS(status, valueNT->getBlockOfRows(0, 1, ReadWriteMode::readWrite, vr));
        algorithmFPType & value = *vr.getBlockPtr();

        UniversalBuffer logLosUniversal = ctx.allocate(idType, n, status);
        DAAL_CHECK_STATUS_VAR(status);
        services::internal::Buffer<algorithmFPType> logLossBuff = logLosUniversal.get<algorithmFPType>();

        value = algorithmFPType(0);
        DAAL_CHECK_STATUS(status, logLoss(yBuff, sigmoidBuf, logLossBuff, n));
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::sum(logLossBuff, value, n));
        value *= -div;

        if (l1reg > 0 || l2reg > 0)
        {
            algorithmFPType reg = algorithmFPType(0);
            // + l1*||Beta|| + l2*||Beta||**2
            // Beta = (B1, B2, ... Bk)
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::regularization(argBuff, nBeta, 1, reg, l1reg, l2reg));

            value += reg;
        }
        DAAL_CHECK_STATUS(status, valueNT->releaseBlockOfRows(vr));
    }

    if (gradientNT)
    {
        DAAL_ASSERT(gradientNT->getNumberOfRows() == nBeta);

        BlockDescriptor<algorithmFPType> gr;
        DAAL_CHECK_STATUS(status, gradientNT->getBlockOfRows(0, nBeta, ReadWriteMode::readWrite, gr));
        daal::services::internal::Buffer<algorithmFPType> gradientBuff = gr.getBuffer();

        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_subSigmoidYUniversal, n));
        services::internal::Buffer<algorithmFPType> subSigmoidYBuff = _subSigmoidYUniversal.get<algorithmFPType>();

        // diff = sigmoid(Xb) - y
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(subVectors);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::subVectors(sigmoidBuf, yBuff, subSigmoidYBuff, n));
        }

        const algorithmFPType coeffBeta = algorithmFPType(2) * l2reg;
        if (l2reg > 0)
        {
            ctx.copy(gradientBuff, 1, argBuff, 1, nBeta - 1, status);
            DAAL_CHECK_STATUS_VAR(status);
            const algorithmFPType zero = algorithmFPType(0);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setElem(0, zero, gradientBuff));
        }

        // gradient = (X^T(sigmoid(Xb) - y)/n + 2*l2*||Beta||
        DAAL_CHECK_STATUS(status, applyGradient(xBuff, subSigmoidYBuff, gradientBuff, div, n, ldX, coeffBeta, offsetX));

        if (interceptFlag)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(interceptCalculate);
            // g[0] = sum(sigmoid(Xb) - y)/n
            algorithmFPType g0 = algorithmFPType(0);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::sum(subSigmoidYBuff, g0, n));
            g0 *= div;
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setElem(0, g0, gradientBuff));
        }

        DAAL_CHECK_STATUS(status, gradientNT->releaseBlockOfRows(gr));
    }

    if (hessianNT)
    {
        DAAL_ASSERT(hessianNT->getNumberOfRows() == nBeta);
        DAAL_ASSERT(hessianNT->getNumberOfColumns() == nBeta);

        BlockDescriptor<algorithmFPType> hr;
        DAAL_CHECK_STATUS(status, hessianNT->getBlockOfRows(0, nBeta, ReadWriteMode::readWrite, hr));
        daal::services::internal::Buffer<algorithmFPType> hessianBuff = hr.getBuffer();

        DAAL_CHECK_STATUS(status, applyHessian(xBuff, sigmoidBuf, n, ldX, hessianBuff, nBeta, offsetX, div));

        if (interceptFlag)
        {
            DAAL_CHECK_STATUS(status, hessianIntercept(xBuff, sigmoidBuf, n, ldX, hessianBuff, nBeta, div));
        }

        if (l2reg > 0)
        {
            DAAL_CHECK_STATUS(status, hessianRegulization(hessianBuff, nBeta, l2reg));
        }
        DAAL_CHECK_STATUS(status, hessianNT->releaseBlockOfRows(hr));
    }

    return status;
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::compute(NumericTable * data, NumericTable * dependentVariables,
                                                                             NumericTable * argument, NumericTable * value, NumericTable * hessian,
                                                                             NumericTable * gradient, NumericTable * nonSmoothTermValue,
                                                                             NumericTable * proximalProjectionNT, NumericTable * lipschitzConstantNT,
                                                                             Parameter * parameter)
{
    services::Status status;
    DAAL_ASSERT(data != nullptr);
    DAAL_ASSERT(parameter != nullptr);
    DAAL_ASSERT(dependentVariables != nullptr);
    DAAL_ASSERT(argument != nullptr);

    const size_t nRows = data->getNumberOfRows();
    const size_t p     = data->getNumberOfColumns();
    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, p, 1);
    const size_t nBeta = p + 1;

    DAAL_ASSERT(argument->getNumberOfColumns() == 1);
    DAAL_ASSERT(argument->getNumberOfRows() == nBeta);

    BlockDescriptor<algorithmFPType> agrBlock;
    DAAL_CHECK_STATUS(status, argument->getBlockOfRows(0, nBeta, ReadWriteMode::readOnly, agrBlock));

    const services::internal::Buffer<algorithmFPType> argBuff = agrBlock.getBuffer();

    NumericTable * ntInd        = parameter->batchIndices.get();
    const algorithmFPType l1reg = parameter->penaltyL1;
    const algorithmFPType l2reg = parameter->penaltyL2;

    if (ntInd == nullptr || (ntInd != nullptr && ntInd->getNumberOfColumns() == nRows))
    {
        BlockDescriptor<algorithmFPType> xBlock;
        BlockDescriptor<algorithmFPType> yBlock;

        DAAL_CHECK_STATUS(status, data->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, yBlock));

        const services::internal::Buffer<algorithmFPType> xBuff = xBlock.getBuffer();
        const services::internal::Buffer<algorithmFPType> yBuff = yBlock.getBuffer();

        const size_t nBatch      = nRows;
        const bool isSourceData  = true;
        const bool interceptFlag = parameter->interceptFlag;

        status = doCompute(nBatch, p, xBuff, yBuff, argBuff, value, gradient, hessian, nonSmoothTermValue, proximalProjectionNT, lipschitzConstantNT,
                           l1reg, l2reg, interceptFlag, isSourceData);

        DAAL_CHECK_STATUS(status, data->releaseBlockOfRows(xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->releaseBlockOfRows(yBlock));
    }
    else
    {
        const size_t nBatch = ntInd->getNumberOfColumns();
        // TODO: if (nBatch == 1)

        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_uX, nBatch * nBeta));
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_uY, nBatch));

        services::internal::Buffer<algorithmFPType> xBuff = _uX.get<algorithmFPType>();
        services::internal::Buffer<algorithmFPType> yBuff = _uY.get<algorithmFPType>();

        const bool isSourceData  = false;
        const bool interceptFlag = false;

        BlockDescriptor<int> rInd;
        DAAL_CHECK_STATUS(status, ntInd->getBlockOfRows(0, 1, ReadWriteMode::readOnly, rInd));
        services::internal::Buffer<int> indBuff = rInd.getBuffer();

        BlockDescriptor<algorithmFPType> xBlock;
        BlockDescriptor<algorithmFPType> yBlock;

        DAAL_CHECK_STATUS(status, data->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, yBlock));

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(getXY);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::getXY(xBlock.getBuffer(), yBlock.getBuffer(), indBuff, xBuff, yBuff, nBatch, p,
                                                                     parameter->interceptFlag));
        }

        DAAL_CHECK_STATUS(status, ntInd->releaseBlockOfRows(rInd));

        status = doCompute(nBatch, p, xBuff, yBuff, argBuff, value, gradient, hessian, nonSmoothTermValue, proximalProjectionNT, lipschitzConstantNT,
                           l1reg, l2reg, interceptFlag, isSourceData);

        DAAL_CHECK_STATUS(status, data->releaseBlockOfRows(xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->releaseBlockOfRows(yBlock));
    }

    DAAL_CHECK_STATUS(status, argument->releaseBlockOfRows(agrBlock));
    return status;
}

} // namespace internal
} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
