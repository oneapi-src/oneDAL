/* file: cross_entropy_loss_dense_default_oneapi_impl.i */
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
//  Implementation of Cross-Entropy Loss algorithm for GPU.
//--
*/

#include "oneapi/math_service_types.h"
#include "oneapi/blas_gpu.h"
#include "cl_kernel/cross_entropy_loss_dense_default.cl"
#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(optimization_solver.cross_entropy_loss.batch.oneapi);

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace cross_entropy_loss
{
namespace internal
{
using namespace daal::oneapi::internal;
using namespace daal::internal;

// Calculate X^T*beta
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::applyBeta(const services::Buffer<algorithmFPType> & x,
                                                                                        const services::Buffer<algorithmFPType> & beta,
                                                                                        services::Buffer<algorithmFPType> & xb, const uint32_t n,
                                                                                        const uint32_t nClasses, const uint32_t ldX,
                                                                                        const uint32_t nBeta, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyBeta);
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, n, nClasses, ldX,
                                           algorithmFPType(1), x, ldX, 0, beta, nBeta, offset, algorithmFPType(0), xb, nClasses, 0);
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::betaIntercept(const services::Buffer<algorithmFPType> & one,
                                                                                            const services::Buffer<algorithmFPType> & arg,
                                                                                            services::Buffer<algorithmFPType> & f, const uint32_t n,
                                                                                            const uint32_t nClasses, const uint32_t nBeta)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(betaIntercept);

    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, n, nClasses, 1,
                                           algorithmFPType(1), one, 1, 0, arg, nBeta, 0, algorithmFPType(1), f, nClasses, 0);
}

// Calculate (y - sigma)^T*X + 2*L2*beta
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::applyGradient(const services::Buffer<algorithmFPType> & x,
                                                                                            const services::Buffer<algorithmFPType> & g,
                                                                                            services::Buffer<algorithmFPType> & gradient,
                                                                                            const algorithmFPType alpha, const uint32_t n,
                                                                                            const uint32_t p, const uint32_t nBeta, uint32_t nClasses,
                                                                                            const algorithmFPType beta, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyGradient);

    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nClasses, p, n, alpha, g,
                                           nClasses, 0, x, p, 0, beta, gradient, nBeta, offset);
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::applyHessian(
    const services::Buffer<algorithmFPType> & x, const services::Buffer<algorithmFPType> & prob, const uint32_t n, const uint32_t p,
    services::Buffer<algorithmFPType> & h, const uint32_t nBeta, const uint32_t nClasses, const uint32_t offset, const algorithmFPType alpha)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyHessian);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "hessian";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(9);
    args.set(0, x, AccessModeIds::read);
    args.set(1, p);
    args.set(2, prob, AccessModeIds::read);
    args.set(3, n);
    args.set(4, h, AccessModeIds::write);
    args.set(5, nBeta);
    args.set(6, nClasses);
    args.set(7, 0);
    args.set(8, alpha);

    KernelRange range(nBeta * nClasses, nBeta * nClasses);
    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::softmax(const services::Buffer<algorithmFPType> & x,
                                                                                      services::Buffer<algorithmFPType> & result, const uint32_t n,
                                                                                      const uint32_t nClasses)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(softmax);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "softmax";
    KernelPtr kernel              = factory.getKernel(kernelName);

    const algorithmFPType expThreshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(4);
    args.set(0, x, AccessModeIds::read);
    args.set(1, result, AccessModeIds::readwrite);
    args.set(2, nClasses);
    args.set(3, expThreshold);

    KernelRange range(n);

    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::softmaxAndUpdateProba(const services::Buffer<algorithmFPType> & x,
                                                                                                    const services::Buffer<algorithmFPType> & y,
                                                                                                    services::Buffer<algorithmFPType> & result,
                                                                                                    const uint32_t n, const uint32_t nClasses)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(softmaxAndUpdateProba);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "softmaxAndUpdateProba";
    KernelPtr kernel              = factory.getKernel(kernelName);

    const algorithmFPType expThreshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(5);
    args.set(0, x, AccessModeIds::read);
    args.set(1, y, AccessModeIds::read);
    args.set(2, result, AccessModeIds::readwrite);
    args.set(3, nClasses);
    args.set(4, expThreshold);

    KernelRange range(n);

    ctx.run(range, kernel, args, &status);

    return status;
}

// resulti = [yi=K]*log(sigmai)
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::crossEntropy(const services::Buffer<algorithmFPType> & y,
                                                                                           const services::Buffer<algorithmFPType> & sigma,
                                                                                           services::Buffer<algorithmFPType> & result,
                                                                                           const uint32_t n, const uint32_t nClasses)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(crossEntropy);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "crossEntropy";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(4);
    args.set(0, y, AccessModeIds::read);
    args.set(1, sigma, AccessModeIds::read);
    args.set(2, result, AccessModeIds::write);
    args.set(3, nClasses);

    KernelRange range(n);

    ctx.run(range, kernel, args, &status);

    return status;
}

// resulti = [yi=K]*log(sigmai)
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::updateProba(const services::Buffer<algorithmFPType> & y,
                                                                                          services::Buffer<algorithmFPType> & sigma, const uint32_t n,
                                                                                          const uint32_t nClasses, const algorithmFPType value)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateProba);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "updateProba";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(4);
    args.set(0, y, AccessModeIds::read);
    args.set(1, sigma, AccessModeIds::readwrite);
    args.set(2, nClasses);
    args.set(3, value);

    KernelRange range(n);
    {
        ctx.run(range, kernel, args, &status);
    }

    return status;
}

template <typename algorithmFPType>
void CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);
    services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_optimization_solver_cross_entropy_loss_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelCrossEntropyLoss, options.c_str());
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::doCompute(
    const uint32_t nBatch, const uint32_t nFeatures, const uint32_t nClasses, const daal::services::Buffer<algorithmFPType> & xBuff,
    const daal::services::Buffer<algorithmFPType> & yBuff, const daal::services::Buffer<algorithmFPType> & argBuff, NumericTable * valueNT,
    NumericTable * gradientNT, NumericTable * hessianNT, NumericTable * nonSmoothTermValueNT, NumericTable * proximalProjectionNT,
    NumericTable * lipschitzConstantNT, const algorithmFPType l1reg, const algorithmFPType l2reg, const bool interceptFlag, const bool isSourceData)
{
    services::Status status;

    ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

    const uint32_t nBeta   = nFeatures + 1;
    const uint32_t ldX     = isSourceData || interceptFlag ? nFeatures : nBeta;
    const uint32_t offsetX = isSourceData || interceptFlag ? 1 : 0;

    const uint32_t n = nBatch;

    if (hessianNT || nonSmoothTermValueNT || proximalProjectionNT || lipschitzConstantNT)
    {
        // TODO
        return services::ErrorMethodNotImplemented;
    }

    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_fUniversal, n * nClasses));
    services::Buffer<algorithmFPType> fBuf = _fUniversal.get<algorithmFPType>();

    //f = X*W + W0
    DAAL_CHECK_STATUS(status, applyBeta(xBuff, argBuff, fBuf, n, nClasses, ldX, nBeta, offsetX));

    if (interceptFlag)
    {
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_oneVector, n));
        services::Buffer<algorithmFPType> oneVectorBuf = _oneVector.get<algorithmFPType>();

        ctx.fill(_oneVector, 1.0, &status);
        DAAL_CHECK_STATUS(status, betaIntercept(oneVectorBuf, argBuff, fBuf, n, nClasses, nBeta));
    }

    const bool isNotOnlyGrad = valueNT || hessianNT;

    services::Buffer<algorithmFPType> softmaxBuf;
    if (isNotOnlyGrad)
    {
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_softmaxUniversal, n * nClasses));
        softmaxBuf = _softmaxUniversal.get<algorithmFPType>();
        DAAL_CHECK_STATUS(status, softmax(fBuf, softmaxBuf, n, nClasses));
    }

    const algorithmFPType div = algorithmFPType(1) / algorithmFPType(n);

    if (valueNT)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(doCompute.valueNT);
        DAAL_ASSERT(valueNT->getNumberOfRows() == 1);

        BlockDescriptor<algorithmFPType> vr;
        DAAL_CHECK_STATUS(status, valueNT->getBlockOfRows(0, 1, ReadWriteMode::readWrite, vr));
        algorithmFPType & value = *vr.getBlockPtr();

        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_crossEntropyUniversal, n));
        services::Buffer<algorithmFPType> crossEntropyBuff = _crossEntropyUniversal.get<algorithmFPType>();

        DAAL_CHECK_STATUS(status, crossEntropy(yBuff, softmaxBuf, crossEntropyBuff, n, nClasses));

        // TODO replace mean
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::sum(crossEntropyBuff, value, n));
        value *= -div;

        if (l1reg > 0 || l2reg > 0)
        {
            algorithmFPType reg = algorithmFPType(0);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::regularization(argBuff, nBeta, nClasses, reg, l1reg, l2reg));

            value += reg;
        }
        DAAL_CHECK_STATUS(status, valueNT->releaseBlockOfRows(vr));
    }

    if (gradientNT)
    {
        DAAL_ASSERT(gradientNT->getNumberOfRows() == nClasses * nBeta);

        BlockDescriptor<algorithmFPType> gr;
        DAAL_CHECK_STATUS(status, gradientNT->getBlockOfRows(0, nClasses * nBeta, ReadWriteMode::readWrite, gr));
        daal::services::Buffer<algorithmFPType> gradientBuff = gr.getBuffer();

        const algorithmFPType zero = algorithmFPType(0);

        // fBuf = fBuf - 1 if y_i == K
        if (isNotOnlyGrad)
        {
            const algorithmFPType minusOne = algorithmFPType(-1);
            DAAL_CHECK_STATUS(status, updateProba(yBuff, softmaxBuf, n, nClasses, minusOne));
        }
        else
        {
            softmaxBuf = fBuf;
            DAAL_CHECK_STATUS(status, softmaxAndUpdateProba(fBuf, yBuff, softmaxBuf, n, nClasses));
        }

        const algorithmFPType coeffBeta = algorithmFPType(2) * l2reg;
        if (l2reg > 0)
        {
            ctx.copy(gradientBuff, 0, argBuff, 0, nBeta * nClasses, &status);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setColElem(0, zero, gradientBuff, nClasses, nBeta));
        }

        // gradient = (X^T(sigmoid(Xb) - y)/n + 2*l2*||Beta||
        DAAL_CHECK_STATUS(status, applyGradient(xBuff, softmaxBuf, gradientBuff, div, n, ldX, nBeta, nClasses, coeffBeta, offsetX));

        if (interceptFlag)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(doCompute.gradientNT.interceptFlag);
            // g[0] = sum(sigmoid(Xb) - y)/n
            DAAL_CHECK_STATUS(status,
                              applyGradient(_oneVector.get<algorithmFPType>(), softmaxBuf, gradientBuff, div, n, 1, nBeta, nClasses, zero, 0u));
        }

        if (hessianNT)
        {
            // fBuf = fBuf + 1 if y_i == K
            const algorithmFPType one = algorithmFPType(1);
            DAAL_CHECK_STATUS(status, updateProba(yBuff, softmaxBuf, n, nClasses, one));
        }
        DAAL_CHECK_STATUS(status, gradientNT->releaseBlockOfRows(gr));
    }

    return status;
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::compute(NumericTable * data, NumericTable * dependentVariables,
                                                                                      NumericTable * argument, NumericTable * value,
                                                                                      NumericTable * hessian, NumericTable * gradient,
                                                                                      NumericTable * nonSmoothTermValue,
                                                                                      NumericTable * proximalProjectionNT,
                                                                                      NumericTable * lipschitzConstantNT, Parameter * parameter)
{
    services::Status status;

    const size_t nRows    = data->getNumberOfRows();
    const size_t p        = data->getNumberOfColumns();
    const size_t nBeta    = p + 1;
    const size_t nClasses = parameter->nClasses;

    DAAL_ASSERT(argument->getNumberOfColumns() == 1);
    DAAL_ASSERT(argument->getNumberOfRows() == nClasses * nBeta);

    BlockDescriptor<algorithmFPType> agrBlock;
    DAAL_CHECK_STATUS(status, argument->getBlockOfRows(0, nClasses * nBeta, ReadWriteMode::readOnly, agrBlock));

    services::Buffer<algorithmFPType> argBuff = agrBlock.getBuffer();

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

        status = doCompute(nBatch, p, nClasses, xBuff, yBuff, argBuff, value, gradient, hessian, nonSmoothTermValue, proximalProjectionNT,
                           lipschitzConstantNT, l1reg, l2reg, interceptFlag, isSourceData);

        DAAL_CHECK_STATUS(status, data->releaseBlockOfRows(xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->releaseBlockOfRows(yBlock));
    }
    else
    {
        const size_t nBatch = ntInd->getNumberOfColumns();

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

        status = doCompute(nBatch, p, nClasses, xBuff, yBuff, argBuff, value, gradient, hessian, nonSmoothTermValue, proximalProjectionNT,
                           lipschitzConstantNT, l1reg, l2reg, interceptFlag, isSourceData);

        DAAL_CHECK_STATUS(status, data->releaseBlockOfRows(xBlock));
        DAAL_CHECK_STATUS(status, dependentVariables->releaseBlockOfRows(yBlock));
    }

    DAAL_CHECK_STATUS(status, argument->releaseBlockOfRows(agrBlock));
    return status;
}

} // namespace internal
} // namespace cross_entropy_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
