/* file: logistic_loss_dense_default_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "cl_kernel/logistic_loss_dense_default.cl"
#include "service_utils.h"
#include "service_math.h"

#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(optimization_solver.logistic_loss.batch.oneapi);

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
using namespace daal::oneapi::internal;
using namespace daal::internal;

// Calculate X^T*beta
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::applyBeta(const services::Buffer<algorithmFPType> & x,
                                                                               const services::Buffer<algorithmFPType> & beta,
                                                                               services::Buffer<algorithmFPType> & xb, const uint32_t n,
                                                                               const uint32_t p, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyBeta);
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans, n, 1, p, algorithmFPType(1), x,
                                           p, 0, beta, 1, offset, algorithmFPType(0), xb, 1, 0);
}

// Calculate X^T*(y - sigma) + 2*L2*beta
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::applyGradient(const services::Buffer<algorithmFPType> & x,
                                                                                   const services::Buffer<algorithmFPType> & sub,
                                                                                   services::Buffer<algorithmFPType> & gradient,
                                                                                   const algorithmFPType alpha, const uint32_t n, const uint32_t p,
                                                                                   const algorithmFPType beta, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyGradient);
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, p, 1, n, alpha, x, p, 0, sub, 1,
                                           0, beta, gradient, 1, offset);
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::applyHessian(const services::Buffer<algorithmFPType> & x,
                                                                                  const services::Buffer<algorithmFPType> & sigma, const uint32_t n,
                                                                                  const uint32_t p, services::Buffer<algorithmFPType> & h,
                                                                                  const uint32_t nBeta, const uint32_t offset,
                                                                                  const algorithmFPType alpha)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyHessian);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "hessian";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(8);
    args.set(0, x, AccessModeIds::read);
    args.set(1, p);
    args.set(2, sigma, AccessModeIds::read);
    args.set(3, n);
    args.set(4, h, AccessModeIds::write);
    args.set(5, nBeta);
    args.set(6, offset);
    args.set(7, alpha);

    KernelRange range(p, p);

    ctx.run(range, kernel, args, &status);

    return services::Status();
}

// ylog(sigm) + (1-y)log(1-sigma)
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::logLoss(const services::Buffer<algorithmFPType> & y,
                                                                             const services::Buffer<algorithmFPType> & sigma,
                                                                             services::Buffer<algorithmFPType> & result, const uint32_t n)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(logLoss);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "logLoss";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(3);
    args.set(0, y, AccessModeIds::read);
    args.set(1, sigma, AccessModeIds::read);
    args.set(2, result, AccessModeIds::write);

    KernelRange range(n);
    ctx.run(range, kernel, args, &status);
    return status;
}

// sigmoid(x) = 1/(1+exp(-x))
// if calculateInverse = true, x[i][1] = 1 - sigmoid(x[i][0])
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::sigmoids(const services::Buffer<algorithmFPType> & x,
                                                                              services::Buffer<algorithmFPType> & result, const uint32_t n,
                                                                              bool calculateInverse)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(sigmoids);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "sigmoid";
    KernelPtr kernel              = factory.getKernel(kernelName);

    const algorithmFPType expThreshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(4);
    args.set(0, x, AccessModeIds::read);
    args.set(1, expThreshold);
    args.set(2, uint32_t(calculateInverse));
    args.set(3, result, AccessModeIds::write);

    KernelRange range(n);
    ctx.run(range, kernel, args, &status);
    return status;
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::betaIntercept(const services::Buffer<algorithmFPType> & arg,
                                                                                   services::Buffer<algorithmFPType> & x, const uint32_t n)
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
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::hessianIntercept(const services::Buffer<algorithmFPType> & x,
                                                                                      const services::Buffer<algorithmFPType> & sigma,
                                                                                      const uint32_t n, const uint32_t p,
                                                                                      services::Buffer<algorithmFPType> & h, const uint32_t nBeta,
                                                                                      const algorithmFPType alpha)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(hessianIntercept);
    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    services::Status status;
    {
        const char * const kernelName = "hessianIntercept";
        KernelPtr kernel              = factory.getKernel(kernelName);

        KernelArguments args(7);
        args.set(0, x, AccessModeIds::read);
        args.set(1, p);
        args.set(2, sigma, AccessModeIds::read);
        args.set(3, n);
        args.set(4, h, AccessModeIds::write);
        args.set(5, nBeta);
        args.set(6, alpha);

        KernelRange range(nBeta);

        ctx.run(range, kernel, args, &status);
    }
    {
        // h[0][0] = alpha*sigma[i]*(1-sima[i])
        algorithmFPType h00           = algorithmFPType(0);
        const char * const kernelName = "hessianInterceptH0";

        KernelPtr kernel = factory.getKernel(kernelName);

        KernelNDRange range(1);

        // TODO replace on min
        // size_t workItemsPerGroup = maxWorkItemSizes1d > maxWorkGroupSize ?
        //     maxWorkGroupSize : maxWorkItemSizes1d;

        size_t workItemsPerGroup = 256;

        const size_t nWorkGroups = HelperObjectiveFunction::getWorkgroupsCount(n, workItemsPerGroup);

        KernelRange localRange(workItemsPerGroup);
        KernelRange globalRange(nWorkGroups * workItemsPerGroup);

        range.local(localRange, &status);
        range.global(globalRange, &status);
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer buffer                            = ctx.allocate(idType, nWorkGroups, &status);
        services::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

        KernelArguments args(3 /*4*/);
        args.set(0, sigma, AccessModeIds::read);
        args.set(1, n);
        args.set(2, reductionBuffer, AccessModeIds::write);
        //args.set(3, LocalBuffer(idType, workItemsPerGroup));

        ctx.run(range, kernel, args, &status);

        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::sumReduction(reductionBuffer, nWorkGroups, h00));

        h00 *= alpha;
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setElem(0, h00, h));
    }
    return services::Status();
}

// ylog(sigm) + (1-y)log(1-sigma)
template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::hessianRegulization(services::Buffer<algorithmFPType> & h, const uint32_t nBeta,
                                                                                         const algorithmFPType l2)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(hessianRegulization);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "hessianRegulization";
    KernelPtr kernel              = factory.getKernel(kernelName);

    const algorithmFPType beta = l2 * algorithmFPType(2);

    KernelArguments args(3);
    args.set(0, h, AccessModeIds::write);
    args.set(1, nBeta);
    args.set(2, beta);

    KernelRange range(nBeta - 1);

    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType>
void LogLossKernelOneAPI<algorithmFPType, defaultDense>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_optimization_solver_logistic_loss_");
    cachekey.add(options);

    options.add(" -D LOCAL_SUM_SIZE=256 ");

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelLogLoss, options.c_str());
}

template <typename algorithmFPType>
services::Status LogLossKernelOneAPI<algorithmFPType, defaultDense>::doCompute(
    const uint32_t nBatch, const uint32_t nFeatures, const daal::services::Buffer<algorithmFPType> & xBuff,
    const daal::services::Buffer<algorithmFPType> & yBuff, const daal::services::Buffer<algorithmFPType> & argBuff, NumericTable * valueNT,
    NumericTable * gradientNT, NumericTable * hessianNT, NumericTable * nonSmoothTermValueNT, NumericTable * proximalProjectionNT,
    NumericTable * lipschitzConstantNT, const algorithmFPType l1reg, const algorithmFPType l2reg, const bool interceptFlag, const bool isSourceData)
{
    services::Status status;

    ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

    const uint32_t nBeta   = nFeatures + 1;
    const uint32_t ldX     = isSourceData ? nFeatures : nBeta;
    const uint32_t offsetX = isSourceData ? 1 : 0;

    const uint32_t n = nBatch;

    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    if (valueNT == nullptr && gradientNT == nullptr && hessianNT == nullptr)
    {
        return services::ErrorMethodNotImplemented;
    }

    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_fUniversal, n));
    services::Buffer<algorithmFPType> fBuf = _fUniversal.get<algorithmFPType>();

    //f = X*b + b0
    DAAL_CHECK_STATUS(status, applyBeta(xBuff, argBuff, fBuf, n, ldX, offsetX));

    if (interceptFlag)
    {
        DAAL_CHECK_STATUS(status, betaIntercept(argBuff, fBuf, n));
    }

    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_sigmoidUniversal, n));
    services::Buffer<algorithmFPType> sigmoidBuf = _sigmoidUniversal.get<algorithmFPType>();

    //s = exp(-f)
    DAAL_CHECK_STATUS(status, sigmoids(fBuf, sigmoidBuf, n));
    const algorithmFPType div = algorithmFPType(1) / algorithmFPType(n);

    if (valueNT)
    {
        DAAL_ASSERT(valueNT->getNumberOfRows() == 1);

        BlockDescriptor<algorithmFPType> vr;
        DAAL_CHECK_STATUS(status, valueNT->getBlockOfRows(0, 1, ReadWriteMode::readWrite, vr));
        algorithmFPType & value = *vr.getBlockPtr();

        UniversalBuffer logLosUniversal               = ctx.allocate(idType, n, &status);
        services::Buffer<algorithmFPType> logLossBuff = logLosUniversal.get<algorithmFPType>();

        value = algorithmFPType(0);
        DAAL_CHECK_STATUS(status, logLoss(yBuff, sigmoidBuf, logLossBuff, n));

        // TODO replace
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
        daal::services::Buffer<algorithmFPType> gradientBuff = gr.getBuffer();

        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_subSigmoidYUniversal, n));
        services::Buffer<algorithmFPType> subSigmoidYBuff = _subSigmoidYUniversal.get<algorithmFPType>();

        // diff = sigmoid(Xb) - y
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(subVectors);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::subVectors(sigmoidBuf, yBuff, subSigmoidYBuff, n));
        }

        const algorithmFPType coeffBeta = algorithmFPType(2) * l2reg;
        if (l2reg > 0)
        {
            ctx.copy(gradientBuff, 1, argBuff, 1, nBeta - 1, &status);
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
        daal::services::Buffer<algorithmFPType> hessianBuff = hr.getBuffer();

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

    const size_t nRows = data->getNumberOfRows();
    const size_t p     = data->getNumberOfColumns();
    const size_t nBeta = p + 1;

    DAAL_ASSERT(argument->getNumberOfColumns() == 1);
    DAAL_ASSERT(argument->getNumberOfRows() == nBeta);

    BlockDescriptor<algorithmFPType> agrBlock;
    DAAL_CHECK_STATUS(status, argument->getBlockOfRows(0, nBeta, ReadWriteMode::readOnly, agrBlock));

    const services::Buffer<algorithmFPType> argBuff = agrBlock.getBuffer();

    NumericTable * ntInd        = parameter->batchIndices.get();
    const algorithmFPType l1reg = parameter->penaltyL1;
    const algorithmFPType l2reg = parameter->penaltyL2;

    if (ntInd == nullptr || (ntInd != nullptr && ntInd->getNumberOfColumns() == nRows))
    {
        BlockDescriptor<algorithmFPType> xBlock;
        BlockDescriptor<algorithmFPType> yBlock;

        DAAL_CHECK_STATUS(status, data->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, yBlock));

        const services::Buffer<algorithmFPType> xBuff = xBlock.getBuffer();
        const services::Buffer<algorithmFPType> yBuff = yBlock.getBuffer();

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

        services::Buffer<algorithmFPType> xBuff = _uX.get<algorithmFPType>();
        services::Buffer<algorithmFPType> yBuff = _uY.get<algorithmFPType>();

        const bool isSourceData  = false;
        const bool interceptFlag = false;

        BlockDescriptor<int> rInd;
        DAAL_CHECK_STATUS(status, ntInd->getBlockOfRows(0, 1, ReadWriteMode::readOnly, rInd));
        services::Buffer<int> indBuff = rInd.getBuffer();

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
