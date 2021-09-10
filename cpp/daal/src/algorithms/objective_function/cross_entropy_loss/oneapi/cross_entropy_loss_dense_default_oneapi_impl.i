/* file: cross_entropy_loss_dense_default_oneapi_impl.i */
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
//  Implementation of Cross-Entropy Loss algorithm for GPU.
//--
*/

#include "src/sycl/math_service_types.h"
#include "src/sycl/blas_gpu.h"
#include "src/algorithms/objective_function/cross_entropy_loss/oneapi/cl_kernel/cross_entropy_loss_dense_default.cl"
#include "src/externals/service_profiler.h"

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
using namespace daal::services::internal::sycl;
using namespace daal::internal;

// Calculate X^T*beta
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::applyBeta(const services::internal::Buffer<algorithmFPType> & x,
                                                                                        const services::internal::Buffer<algorithmFPType> & beta,
                                                                                        services::internal::Buffer<algorithmFPType> & xb,
                                                                                        const uint32_t n, const uint32_t nClasses, const uint32_t ldX,
                                                                                        const uint32_t nBeta, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyBeta);
    DAAL_ASSERT(x.size() >= size_t(n) * size_t(ldX)); // overflows checked in the algorithm
    DAAL_ASSERT(beta.size() >= size_t(offset) + size_t(ldX) * size_t(nClasses));
    DAAL_ASSERT(xb.size() >= size_t(n) * size_t(nClasses));

    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, n, nClasses, ldX,
                                           algorithmFPType(1), x, ldX, 0, beta, nBeta, offset, algorithmFPType(0), xb, nClasses, 0);
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::betaIntercept(const services::internal::Buffer<algorithmFPType> & one,
                                                                                            const services::internal::Buffer<algorithmFPType> & arg,
                                                                                            services::internal::Buffer<algorithmFPType> & f,
                                                                                            const uint32_t n, const uint32_t nClasses,
                                                                                            const uint32_t nBeta)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(betaIntercept);
    DAAL_ASSERT(one.size() >= size_t(n));
    DAAL_ASSERT(arg.size() >= size_t(nClasses));
    DAAL_ASSERT(f.size() >= size_t(n) * size_t(nClasses)); // overflows checked in the algorithm

    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, n, nClasses, 1,
                                           algorithmFPType(1), one, 1, 0, arg, nBeta, 0, algorithmFPType(1), f, nClasses, 0);
}

// Calculate (y - sigma)^T*X + 2*L2*beta
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::applyGradient(const services::internal::Buffer<algorithmFPType> & x,
                                                                                            const services::internal::Buffer<algorithmFPType> & g,
                                                                                            services::internal::Buffer<algorithmFPType> & gradient,
                                                                                            const algorithmFPType alpha, const uint32_t n,
                                                                                            const uint32_t p, const uint32_t nBeta, uint32_t nClasses,
                                                                                            const algorithmFPType beta, const uint32_t offset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(applyGradient);
    DAAL_ASSERT(g.size() >= size_t(n) * size_t(nClasses));
    DAAL_ASSERT(x.size() >= size_t(n) * size_t(p));
    DAAL_ASSERT(gradient.size() >= size_t(offset) + size_t(p) * size_t(nClasses)); // overflows checked in the algorithm

    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nClasses, p, n, alpha, g,
                                           nClasses, 0, x, p, 0, beta, gradient, nBeta, offset);
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::softmax(const services::internal::Buffer<algorithmFPType> & x,
                                                                                      services::internal::Buffer<algorithmFPType> & result,
                                                                                      const uint32_t n, const uint32_t nClasses)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(softmax);

    DAAL_ASSERT(x.size() >= size_t(n) * size_t(nClasses)); // overflows checked in the algorithm
    DAAL_ASSERT(result.size() >= size_t(n) * size_t(nClasses));

    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "softmax";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    const algorithmFPType expThreshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, x, AccessModeIds::read);
    args.set(1, result, AccessModeIds::readwrite);
    args.set(2, nClasses);
    args.set(3, expThreshold);

    KernelRange range(n);

    ctx.run(range, kernel, args, status);

    return status;
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::softmaxAndUpdateProba(
    const services::internal::Buffer<algorithmFPType> & x, const services::internal::Buffer<algorithmFPType> & y,
    services::internal::Buffer<algorithmFPType> & result, const uint32_t n, const uint32_t nClasses)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(softmaxAndUpdateProba);

    DAAL_ASSERT(x.size() >= size_t(n) * size_t(nClasses)); // overflows checked in the algorithm
    DAAL_ASSERT(result.size() >= size_t(n) * size_t(nClasses));
    DAAL_ASSERT(y.size() >= n);

    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "softmaxAndUpdateProba";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    const algorithmFPType expThreshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, x, AccessModeIds::read);
    args.set(1, y, AccessModeIds::read);
    args.set(2, result, AccessModeIds::readwrite);
    args.set(3, nClasses);
    args.set(4, expThreshold);

    KernelRange range(n);

    ctx.run(range, kernel, args, status);

    return status;
}

// resulti = [yi=K]*log(sigmai)
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::crossEntropy(const services::internal::Buffer<algorithmFPType> & y,
                                                                                           const services::internal::Buffer<algorithmFPType> & sigma,
                                                                                           services::internal::Buffer<algorithmFPType> & result,
                                                                                           const uint32_t n, const uint32_t nClasses)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(crossEntropy);
    services::Status status;

    DAAL_ASSERT(sigma.size() == size_t(n) * size_t(nClasses)); // overflows checked in the algorithm
    DAAL_ASSERT(result.size() == n);
    DAAL_ASSERT(y.size() == n);

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "crossEntropy";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, y, AccessModeIds::read);
    args.set(1, sigma, AccessModeIds::read);
    args.set(2, result, AccessModeIds::write);
    args.set(3, nClasses);

    KernelRange range(n);

    ctx.run(range, kernel, args, status);

    return status;
}

// resulti = [yi=K]*log(sigmai)
template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::updateProba(const services::internal::Buffer<algorithmFPType> & y,
                                                                                          services::internal::Buffer<algorithmFPType> & sigma,
                                                                                          const uint32_t n, const uint32_t nClasses,
                                                                                          const algorithmFPType value)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateProba);
    services::Status status;

    DAAL_ASSERT(sigma.size() == size_t(n) * size_t(nClasses)); // overflows checked in the algorithm
    DAAL_ASSERT(y.size() == n);

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "updateProba";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, y, AccessModeIds::read);
    args.set(1, sigma, AccessModeIds::readwrite);
    args.set(2, nClasses);
    args.set(3, value);

    KernelRange range(n);
    {
        ctx.run(range, kernel, args, status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);
    services::Status status;
    services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_optimization_solver_cross_entropy_loss_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelCrossEntropyLoss, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType>
services::Status CrossEntropyLossKernelOneAPI<algorithmFPType, defaultDense>::doCompute(
    const uint32_t nBatch, const uint32_t nFeatures, const uint32_t nClasses, const daal::services::internal::Buffer<algorithmFPType> & xBuff,
    const daal::services::internal::Buffer<algorithmFPType> & yBuff, const daal::services::internal::Buffer<algorithmFPType> & argBuff,
    NumericTable * valueNT, NumericTable * gradientNT, NumericTable * hessianNT, NumericTable * nonSmoothTermValueNT,
    NumericTable * proximalProjectionNT, NumericTable * lipschitzConstantNT, const algorithmFPType l1reg, const algorithmFPType l2reg,
    const bool interceptFlag, const bool isSourceData)
{
    services::Status status;

    ExecutionContextIface & ctx = services::internal::getDefaultContext();

    const uint32_t nBeta = nFeatures + 1;
    DAAL_ASSERT(nBeta > nFeatures);

    const uint32_t ldX     = isSourceData || interceptFlag ? nFeatures : nBeta;
    const uint32_t offsetX = isSourceData || interceptFlag ? 1 : 0;

    const uint32_t n = nBatch;

    if (hessianNT || nonSmoothTermValueNT || proximalProjectionNT || lipschitzConstantNT)
    {
        return services::ErrorMethodNotImplemented;
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, n, nClasses);

    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_fUniversal, n * nClasses));
    services::internal::Buffer<algorithmFPType> fBuf = _fUniversal.get<algorithmFPType>();

    //f = X*W + W0
    DAAL_CHECK_STATUS(status, applyBeta(xBuff, argBuff, fBuf, n, nClasses, ldX, nBeta, offsetX));

    if (interceptFlag)
    {
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_oneVector, n));
        services::internal::Buffer<algorithmFPType> oneVectorBuf = _oneVector.get<algorithmFPType>();

        ctx.fill(_oneVector, 1.0, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_STATUS(status, betaIntercept(oneVectorBuf, argBuff, fBuf, n, nClasses, nBeta));
    }

    const bool isNotOnlyGrad = valueNT != nullptr || hessianNT != nullptr;

    services::internal::Buffer<algorithmFPType> softmaxBuf;
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
        services::internal::Buffer<algorithmFPType> crossEntropyBuff = _crossEntropyUniversal.get<algorithmFPType>();

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
        daal::services::internal::Buffer<algorithmFPType> gradientBuff = gr.getBuffer();

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
            // overflow checked in compute()
            ctx.copy(gradientBuff, 0, argBuff, 0, nBeta * nClasses, status);
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
    DAAL_ASSERT(data != nullptr);
    DAAL_ASSERT(parameter != nullptr);
    DAAL_ASSERT(dependentVariables != nullptr);
    DAAL_ASSERT(argument != nullptr);

    const size_t nRows = data->getNumberOfRows();
    const size_t p     = data->getNumberOfColumns();
    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, p, 1);
    const size_t nBeta    = p + 1;
    const size_t nClasses = parameter->nClasses;

    DAAL_ASSERT(argument->getNumberOfColumns() == 1);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nBeta, nClasses);
    DAAL_ASSERT(argument->getNumberOfRows() == nClasses * nBeta);

    BlockDescriptor<algorithmFPType> agrBlock;
    DAAL_CHECK_STATUS(status, argument->getBlockOfRows(0, nClasses * nBeta, ReadWriteMode::readOnly, agrBlock));

    services::internal::Buffer<algorithmFPType> argBuff = agrBlock.getBuffer();

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
