/* file: pca_transform_dense_default_batch_oneapi_impl.i */
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
//  Common functions of PCA transformation on GPU
//--
*/

#ifndef __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_ONEAPI_IMPL_I__
#define __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_ONEAPI_IMPL_I__

#include "cl_kernels/pca_transform_cl_kernels.cl"
#include <iostream>
using namespace daal::services;
using namespace daal::oneapi::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace oneapi
{
namespace internal
{

using namespace daal::oneapi::internal;

template<typename algorithmFPType, transform::Method method>
void TransformKernelOneAPI<algorithmFPType, method>::computeTransformedBlock
        (const uint32_t numRows, const uint32_t numFeatures, const uint32_t numComponents,
         UniversalBuffer & dataBlock, UniversalBuffer & eigenvectors,
         const services::Buffer<algorithmFPType> & resultBlock)
{
    BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, numComponents, numRows, numFeatures,
                                    1.0, eigenvectors, numFeatures, 0, dataBlock, numFeatures, 0, 0.0, resultBlock, numComponents, 0);
}


template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::computeInvSigmas(NumericTable* variances,
                                                                                  const services::Buffer<algorithmFPType> & invSigmas,
                                                                                  const uint32_t numFeatures)
{
    services::Status status;

    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_pca_transform");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), pca_transform_cl_kernels, build_options.c_str());

    const char * const computeInvSigmasKernel = "computeInvSigmas";
    KernelPtr kernel = factory.getKernel(computeInvSigmasKernel);
    BlockDescriptor<algorithmFPType> varBlock;
    variances->getBlockOfRows(0, numFeatures, readOnly, varBlock);

    KernelArguments args(2);
    args.set(0, varBlock.getBuffer(), AccessModeIds::read);
    args.set(1, invSigmas, AccessModeIds::write);
    KernelRange range(numFeatures);
    ctx.run(range, kernel, args, &status);
    variances->releaseBlockOfRows(varBlock);
    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::normalize(UniversalBuffer & copyBlock,
                                                                           UniversalBuffer & rawMeans,
                                                                           UniversalBuffer & invSigmas,
                                                                           uint32_t numMeans,
                                                                           uint32_t numInvSigmas,
                                                                           const uint32_t numFeatures,
                                                                           const uint32_t numVectors)
{
    services::Status status;

    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_pca_transform");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), pca_transform_cl_kernels, build_options.c_str());

    const char * const normalizeKernel = "normalize";
    KernelPtr kernel = factory.getKernel(normalizeKernel);

    const unsigned int workItemsPerGroup = (numFeatures > maxWorkItemsPerGroup) ? maxWorkItemsPerGroup : numFeatures;
    KernelArguments args(7);
    args.set(0, copyBlock, AccessModeIds::readwrite);
    args.set(1, rawMeans, AccessModeIds::read);
    args.set(2, invSigmas, AccessModeIds::read);
    args.set(3, numMeans);
    args.set(4, numInvSigmas);
    args.set(5, maxWorkItemsPerGroup);
    args.set(6, numFeatures);

    KernelRange local_range(workItemsPerGroup);
    KernelRange global_range(workItemsPerGroup * numVectors);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    ctx.run(range, kernel, args, &status);

    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::whitening(const services::Buffer<algorithmFPType> & transformedBlock,
                                                                           UniversalBuffer & invEigenvalues,
                                                                           const uint32_t numComponents,
                                                                           const uint32_t numVectors)
{
    services::Status status;

    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_pca_transform");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), pca_transform_cl_kernels, build_options.c_str());

    const char * const whiteningKernel = "whitening";
    KernelPtr kernel = factory.getKernel(whiteningKernel);

    const unsigned int workItemsPerGroup = (numComponents > maxWorkItemsPerGroup) ? maxWorkItemsPerGroup : numComponents;
    KernelArguments args(4);
    args.set(0, transformedBlock, AccessModeIds::readwrite);
    args.set(1, invEigenvalues, AccessModeIds::read);
    args.set(2, maxWorkItemsPerGroup);
    args.set(3, numComponents);

    KernelRange local_range(workItemsPerGroup);
    KernelRange global_range(workItemsPerGroup * numVectors);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    ctx.run(range, kernel, args, &status);

    return status;
}


template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::allocateBuffer(ExecutionContextIface& context,
                                                                                UniversalBuffer& returnBuffer,
                                                                                uint32_t bufferSize)
{
    services::Status status;

    const algorithmFPType zero = 0.0;
    returnBuffer = context.allocate(TypeIds::id<algorithmFPType>(), bufferSize, &status);
    DAAL_CHECK_STATUS_VAR(status);
    context.fill(returnBuffer, zero, &status);

    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::copyBuffer(ExecutionContextIface& context,
                                                                            UniversalBuffer& returnBuffer,
                                                                            NumericTable* data,
                                                                            uint32_t nRows, uint32_t nCols)

{
    services::Status status;

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS(status, data->getBlockOfRows(0, nRows, ReadWriteMode::readOnly, dataBlock));
    context.copy(returnBuffer, 0, dataBlock.getBuffer(), 0, nRows * nCols, &status);
    data->releaseBlockOfRows(dataBlock);

    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::copyBufferByRef(ExecutionContextIface& context,
                                                                                 UniversalBuffer& returnBuffer,
                                                                                 NumericTable& data,
                                                                                 uint32_t nRows, uint32_t nCols)

{
    services::Status status;

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS(status, data.getBlockOfRows(0, nRows, ReadWriteMode::readOnly, dataBlock));
    context.copy(returnBuffer, 0, dataBlock.getBuffer(), 0, nRows * nCols, &status);
    data.releaseBlockOfRows(dataBlock);

    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::compute(NumericTable& data, NumericTable& eigenvectors,
                                                                         NumericTable *pMeans, NumericTable *pVariances,
                                                                         NumericTable *pEigenvalues, NumericTable &transformedData)
{
    services::Status status;
    ExecutionContextIface &ctx = services::Environment::getInstance()->getDefaultExecutionContext();

    const uint32_t numVectors  = data.getNumberOfRows();
    const uint32_t numFeatures = data.getNumberOfColumns();
    const uint32_t numComponents = transformedData.getNumberOfColumns();

    const algorithmFPType zero = 0.0;

    /* Calculating invSigmas and invEigenValues*/
    UniversalBuffer invSigmas;
    DAAL_CHECK_STATUS(status, allocateBuffer(ctx, invSigmas, numFeatures));

    uint32_t numInvSigmas = 0;
    if (pVariances != nullptr)
    {
        numInvSigmas = numFeatures;
        DAAL_CHECK_STATUS(status, computeInvSigmas(pVariances, invSigmas.template get<algorithmFPType>(), numFeatures));
    }

    UniversalBuffer invEigenvalues;
    DAAL_CHECK_STATUS(status, allocateBuffer(ctx, invEigenvalues, numFeatures));

    if (pEigenvalues != nullptr)
    {
        DAAL_CHECK_STATUS(status, computeInvSigmas(pEigenvalues, invEigenvalues.template get<algorithmFPType>(), numComponents));
    }

    UniversalBuffer rawMeans;
    DAAL_CHECK_STATUS(status, allocateBuffer(ctx, rawMeans, numFeatures));
    uint32_t numMeans = 0;

    if (pMeans != nullptr)
    {
        numMeans = numFeatures;
        DAAL_CHECK_STATUS(status, copyBuffer(ctx, rawMeans, pMeans, numMeans, 1));
    }

    bool isWhitening = pEigenvalues != nullptr;
    bool isNormalize = pMeans != nullptr || pVariances != nullptr;

    SafeStatus safeStat;

    auto copyBlock = ctx.allocate(TypeIds::id<algorithmFPType>(), numVectors * numFeatures, &status);
    DAAL_CHECK_STATUS(status, copyBufferByRef(ctx, copyBlock, data, numVectors, numFeatures));

    if (isNormalize)
    {
        DAAL_CHECK_STATUS(status, normalize(copyBlock, rawMeans, invSigmas, numMeans, numInvSigmas,
                          numFeatures, numVectors));
    }

    /* Retrieve data associated with coefficients */

    auto basis = ctx.allocate(TypeIds::id<algorithmFPType>(), numComponents * numFeatures, &status);
    DAAL_CHECK_STATUS(status, copyBufferByRef(ctx, basis, eigenvectors, numComponents, numFeatures));

    BlockDescriptor<algorithmFPType> transformedBlock;
    DAAL_CHECK_STATUS(status, transformedData.getBlockOfRows(0, transformedData.getNumberOfRows(), ReadWriteMode::readWrite, transformedBlock));


    computeTransformedBlock(numVectors, numFeatures, numComponents, copyBlock,
       basis, transformedBlock.getBuffer());

    /* compute whitening to unit variance of transformed data if required */
    if(isWhitening)
    {
        DAAL_CHECK_STATUS(status, whitening(transformedBlock.getBuffer(), invEigenvalues, numComponents, numVectors));
    }

    transformedData.releaseBlockOfRows(transformedBlock);
    return safeStat.detach();
} /* void TransformKernelOneAPI<algorithmFPType, defaultDense>::compute */

} /* namespace internal */
} /* namespace oneapi */
} /* namespace transform */
} /* namespace pca */
} /* namespace algorithms */
} /* namespace daal */

#endif
