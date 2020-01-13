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

using namespace daal::services;
using namespace daal::internal;
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
        (DAAL_INT numRows, DAAL_INT numFeatures, DAAL_INT numComponents,
         UniversalBuffer & dataBlock, UniversalBuffer & eigenvectors,
         services::Buffer<algorithmFPType> resultBlock)
{
    BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, numComponents, numRows, numFeatures,
        1.0, eigenvectors, numFeatures, 0, dataBlock, numFeatures, 0, 0.0, resultBlock, numComponents, 0);

}


template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::computeInvSigmas(ExecutionContextIface& context,
                                                                                  const KernelPtr & computeInvSigmasKernel,
                                                                                  NumericTable* variances,
                                                                                  const services::Buffer<algorithmFPType> & invSigmas,
                                                                                  const size_t numFeatures)
{
    services::Status status;

    BlockDescriptor<algorithmFPType> varBlock;
    variances->getBlockOfRows(0, numFeatures, readOnly, varBlock);

    KernelArguments args(2);
    args.set(0, varBlock.getBuffer(), AccessModeIds::read);
    args.set(1, invSigmas, AccessModeIds::write);
    KernelRange range(numFeatures);
    context.run(range, computeInvSigmasKernel, args, &status);
    variances->releaseBlockOfRows(varBlock);                            
    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::normalize(ExecutionContextIface& context,
                                                                           const KernelPtr & normalizeKernel,
                                                                           UniversalBuffer & copyBlock,
                                                                           UniversalBuffer & rawMeans,
                                                                           UniversalBuffer & invSigmas,
                                                                           size_t numMeans,
                                                                           size_t numInvSigmas,
                                                                           const unsigned int maxWorkItemsPerGroup,
                                                                           const size_t numFeatures,
                                                                           const unsigned int workItemsPerGroup,
                                                                           uint32_t numVectors)
{
    services::Status status;

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

    context.run(range, normalizeKernel, args, &status);

    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::whitening(ExecutionContextIface& context,
                                                                           const KernelPtr & whiteningKernel,
                                                                           UniversalBuffer transformedBlock,
                                                                           UniversalBuffer invEigenvalues,
                                                                           uint32_t numComponents,
                                                                           uint32_t numVectors)
{
    services::Status status;

    KernelArguments args(2);
    args.set(0, transformedBlock, AccessModeIds::readwrite);
    args.set(1, invEigenvalues, AccessModeIds::read);

    KernelRange local_range(numComponents);
    KernelRange global_range(numVectors * numComponents);

    KernelNDRange range(1);
    range.global(global_range, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(local_range, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, whiteningKernel, args, &status);

    return status;
}

template<typename algorithmFPType, transform::Method method>
services::Status TransformKernelOneAPI<algorithmFPType, method>::compute(NumericTable& data, NumericTable& eigenvectors,
                                                                         NumericTable *pMeans, NumericTable *pVariances,
                                                                         NumericTable *pEigenvalues, NumericTable &transformedData)
{
    services::Status status;

    ExecutionContextIface &ctx = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface &factory = ctx.getClKernelFactory();

    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_pca_transform");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), pca_transform_cl_kernels, build_options.c_str());

    DAAL_INT numVectors  = data.getNumberOfRows();
    DAAL_INT numFeatures = data.getNumberOfColumns();
    DAAL_INT numComponents = transformedData.getNumberOfColumns();

    const algorithmFPType zero = 0.0;

    /* Calculating invSigmas and invEigenValues*/
    auto computeInvSigmasKernel = factory.getKernel("computeInvSigmas");
    auto invSigmas = ctx.allocate(TypeIds::id<algorithmFPType>(), numFeatures, &status);
    DAAL_CHECK_STATUS_VAR(status);
    ctx.fill(invSigmas, zero, &status);
    DAAL_CHECK_STATUS_VAR(status);
    size_t numInvSigmas = 0;
    if (pVariances != nullptr)
    {
        numInvSigmas = numFeatures;
        DAAL_CHECK_STATUS(status, computeInvSigmas(ctx, computeInvSigmasKernel, pVariances, invSigmas.template get<algorithmFPType>(), numFeatures));
    }

    auto invEigenvalues  = ctx.allocate(TypeIds::id<algorithmFPType>(), numComponents, &status);
    DAAL_CHECK_STATUS_VAR(status);
    ctx.fill(invEigenvalues, zero, &status);
    size_t numEigValues = 0;
    if (pEigenvalues != nullptr)
    {
        DAAL_CHECK_STATUS(status, computeInvSigmas(ctx, computeInvSigmasKernel, pEigenvalues, invEigenvalues.template get<algorithmFPType>(),
                          numComponents));
    }
    size_t numMeans = 0;

    auto rawMeans = ctx.allocate(TypeIds::id<algorithmFPType>(), numFeatures, &status);
    ctx.fill(rawMeans, zero, &status);
    if (pMeans != nullptr)
    {
        numMeans = numFeatures;
        BlockDescriptor<algorithmFPType> meansBlock;
        DAAL_CHECK_STATUS(status, pMeans->getBlockOfRows(0, numMeans, ReadWriteMode::readOnly, meansBlock));

        ctx.copy(rawMeans, 0, meansBlock.getBuffer(), 0, numMeans, &status);
        DAAL_CHECK_STATUS_VAR(status);
        pMeans->releaseBlockOfRows(meansBlock);
    }

    bool isWhitening = pEigenvalues != nullptr;
    bool isNormalize = pMeans != nullptr || pVariances != nullptr;

    SafeStatus safeStat;

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS(status, data.getBlockOfRows(0, numVectors, ReadWriteMode::readOnly, dataBlock));

    auto copyBlock = ctx.allocate(TypeIds::id<algorithmFPType>(), numVectors * numFeatures, &status);
    ctx.copy(copyBlock, 0, dataBlock.getBuffer(), 0, numVectors * numFeatures, &status);
    DAAL_CHECK_STATUS_VAR(status);
    data.releaseBlockOfRows(dataBlock);

    const unsigned int maxWorkItemsPerGroup = 256;
    unsigned int workItemsPerGroup = (numFeatures > maxWorkItemsPerGroup) ? maxWorkItemsPerGroup : numFeatures;

    if (isNormalize)
    {
        auto normalizeKernel = factory.getKernel("normalize");

        DAAL_CHECK_STATUS(status, normalize(ctx, normalizeKernel, copyBlock, rawMeans, invSigmas, numMeans, numInvSigmas,
                          maxWorkItemsPerGroup, numFeatures, workItemsPerGroup, numVectors));
    }

    /* Retrieve data associated with coefficients */
    BlockDescriptor<algorithmFPType> eigenvectorsBlock;
    DAAL_CHECK_STATUS(status, eigenvectors.getBlockOfRows(0, numComponents, ReadWriteMode::readOnly, eigenvectorsBlock));
    auto basis = ctx.allocate(TypeIds::id<algorithmFPType>(), numComponents * numFeatures, &status);
    DAAL_CHECK_STATUS_VAR(status);
    ctx.copy(basis, 0, eigenvectorsBlock.getBuffer(), 0, numComponents * numFeatures, &status);
    DAAL_CHECK_STATUS_VAR(status);

    eigenvectors.releaseBlockOfRows(eigenvectorsBlock);

    BlockDescriptor<algorithmFPType> transformedBlock;
    DAAL_CHECK_STATUS(status, transformedData.getBlockOfRows(0, transformedData.getNumberOfRows(), ReadWriteMode::readWrite, transformedBlock));

    computeTransformedBlock(numVectors, numFeatures, numComponents, copyBlock,
        basis, transformedBlock.getBuffer());

    /* compute whitening to unit variance of transformed data if required */
    if(isWhitening)
    {

        auto whiteningKernel = factory.getKernel("whitening");

        DAAL_CHECK_STATUS(status, whitening(ctx, whiteningKernel, transformedBlock.getBuffer(), invEigenvalues, numComponents, numVectors));
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
