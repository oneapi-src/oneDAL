/* file: select_indexed.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "services/daal_defines.h"
#include "src/sycl/select_indexed.h"
#include "src/sycl/cl_kernels/select_indexed.cl"
#include "services/internal/execution_context.h"
#include "src/externals/service_rng.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "services/daal_string.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_profiler.h"
#include <iostream>
using namespace daal::data_management;
using namespace daal::services::internal;

constexpr uint32_t maxInt32AsUint32T = static_cast<uint32_t>(daal::services::internal::MaxVal<int32_t>::get());

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace selection
{
services::Status runQuickSelectSimd(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & dataVectors,
                                    const UniversalBuffer & indexVectors, const UniversalBuffer & rndSeq, uint32_t nRndSeq, uint32_t nK,
                                    uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset,
                                    QuickSelectIndexed::Result & result)
{
    services::Status status;

    DAAL_ASSERT(nRndSeq <= maxInt32AsUint32T && nRndSeq > 0);
    DAAL_ASSERT(vectorSize <= maxInt32AsUint32T);
    DAAL_ASSERT(vectorOffset <= maxInt32AsUint32T && vectorOffset >= vectorSize);
    DAAL_ASSERT(lastVectorSize <= vectorSize);
    DAAL_ASSERT(nK <= lastVectorSize);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, vectorOffset, (nVectors - 1));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, nVectors, nK);
    DAAL_OVERFLOW_CHECK_BY_ADDING(int32_t, vectorOffset * (nVectors - 1), lastVectorSize);

    if (dataVectors.type() == TypeIds::float32)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(dataVectors, float, vectorOffset *(nVectors - 1) + lastVectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.values, float, nVectors * nK);
    }
    else if (dataVectors.type() == TypeIds::float64)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(dataVectors, double, vectorOffset *(nVectors - 1) + lastVectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.values, double, nVectors * nK);
    }
    else
    {
        return services::Status(ErrorDataTypeNotSupported);
    }
    DAAL_ASSERT_UNIVERSAL_BUFFER(indexVectors, int, vectorOffset *(nVectors - 1) + lastVectorSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(result.indices, int, nVectors * nK);

    auto func_kernel = kernelFactory.getKernel("quick_select_group", status);
    DAAL_CHECK_STATUS_VAR(status);

    const uint32_t maxWorkItemsPerGroup = 16;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(10, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, dataVectors, AccessModeIds::read);
    args.set(1, indexVectors, AccessModeIds::read);
    args.set(2, result.values, AccessModeIds::write);
    args.set(3, result.indices, AccessModeIds::write);
    args.set(4, rndSeq, AccessModeIds::read);
    args.set(5, static_cast<int32_t>(nRndSeq));
    args.set(6, static_cast<int32_t>(vectorSize));
    args.set(7, static_cast<int32_t>(lastVectorSize));
    args.set(8, static_cast<int32_t>(nK));
    args.set(9, static_cast<int32_t>(vectorOffset));

    context.run(range, func_kernel, args, status);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

services::Status runDirectSelectSimd(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & dataVectors,
                                     uint32_t nK, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset,
                                     QuickSelectIndexed::Result & result)
{
    services::Status status;
    DAAL_ASSERT(vectorSize <= maxInt32AsUint32T);
    DAAL_ASSERT(vectorOffset <= maxInt32AsUint32T && vectorOffset >= vectorSize);
    DAAL_ASSERT(lastVectorSize <= vectorSize);
    DAAL_ASSERT(nK <= lastVectorSize);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, vectorOffset, (nVectors - 1));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, nVectors, nK);
    DAAL_OVERFLOW_CHECK_BY_ADDING(int32_t, vectorOffset * (nVectors - 1), lastVectorSize);

    if (dataVectors.type() == TypeIds::float32)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(dataVectors, float, vectorOffset *(nVectors - 1) + lastVectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.values, float, nVectors * nK);
    }
    else if (dataVectors.type() == TypeIds::float64)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(dataVectors, double, vectorOffset *(nVectors - 1) + lastVectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.values, double, nVectors * nK);
    }
    else
    {
        return services::Status(ErrorDataTypeNotSupported);
    }
    DAAL_ASSERT_UNIVERSAL_BUFFER(result.indices, int, nVectors * nK);

    auto func_kernel = kernelFactory.getKernel("direct_select_group", status);
    DAAL_CHECK_STATUS_VAR(status);

    const uint32_t maxWorkItemsPerGroup = 16;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(7, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, dataVectors, AccessModeIds::read);
    args.set(1, result.values, AccessModeIds::write);
    args.set(2, result.indices, AccessModeIds::write);
    args.set(3, static_cast<int32_t>(vectorSize));
    args.set(4, static_cast<int32_t>(lastVectorSize));
    args.set(5, static_cast<int32_t>(vectorOffset));
    if (dataVectors.type() == TypeIds::float32)
    {
        args.set(6, FLT_MAX);
    }
    else
    {
        args.set(6, DBL_MAX);
    }

    context.run(range, func_kernel, args, status);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

services::Status SelectIndexed::convertIndicesToLabels(const UniversalBuffer & indices, const UniversalBuffer & labels, uint32_t nVectors,
                                                       uint32_t vectorSize, uint32_t vectorOffset)
{
    Status status;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, vectorOffset, (nVectors - 1));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, nVectors, vectorSize);
    DAAL_OVERFLOW_CHECK_BY_ADDING(int32_t, vectorOffset * (nVectors - 1), vectorSize);

    DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int, nVectors * vectorSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(labels, int, vectorOffset *(nVectors - 1) + vectorSize);

    auto index2labels = labels.template get<int>().toHost(ReadWriteMode::readOnly, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto index2labelsPtr = index2labels.get();
    auto outIndex        = indices.template get<int>().toHost(ReadWriteMode::readWrite, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto outIndexPtr = outIndex.get();
    for (size_t vec = 0; vec < nVectors; vec++)
    {
        for (size_t k = 0; k < vectorSize; k++)
        {
            int index                         = outIndexPtr[vec * vectorSize + k];
            outIndexPtr[vec * vectorSize + k] = index2labelsPtr[vec * vectorOffset + index];
        }
    }
    return status;
}

services::Status QuickSelectIndexed::buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId)
{
    services::Status status;
    services::String fptypeName = getKeyFPType(vectorTypeId);
    auto buildOptions           = fptypeName;
    buildOptions.add("-cl-std=CL1.2 ");

    services::String cachekey("__daal_oneapi_internal_qselect_indexed_");
    cachekey.add(fptypeName);
    cachekey.add(buildOptions);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), quick_select_simd, buildOptions.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

SelectIndexed::Result & QuickSelectIndexed::selectIndices(const UniversalBuffer & dataVectors, const UniversalBuffer & tempIndices,
                                                          const UniversalBuffer & rndSeq, uint32_t nRndSeq, uint32_t nK, uint32_t nVectors,
                                                          uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset, Result & result,
                                                          services::Status & status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(QuickSelectIndexed.select);

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    status |= buildProgram(kernelFactory, dataVectors.type());
    if (!status.ok())
    {
        return result;
    }
    status |= runQuickSelectSimd(context, kernelFactory, dataVectors, tempIndices, rndSeq, nRndSeq, nK, nVectors, vectorSize, lastVectorSize,
                                 vectorOffset, result);
    return result;
}

Status QuickSelectIndexed::adjustIndexBuffer(uint32_t number, uint32_t size)
{
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, size, number);
    uint32_t newSize = size * number;
    Status status;
    if (_indexSize < newSize)
    {
        auto & context = Environment::getInstance()->getDefaultExecutionContext();
        _indices       = context.allocate(TypeIds::id<int>(), newSize, status);
        DAAL_CHECK_STATUS_VAR(status);
        _indexSize = newSize;
    }
    return status;
}

Status QuickSelectIndexed::init(Params & par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.RNG);
    Status status;
    _nRndSeq        = (par.dataSize > _maxSeqLength || par.dataSize < 2) ? _maxSeqLength : par.dataSize;
    auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(&(*par.engine));
    if (!engineImpl)
    {
        return Status(ErrorIncorrectEngineParameter);
    }
    size_t numbers[_maxSeqLength];
    daal::internal::RNGsInst<size_t, DAAL_BASE_CPU> rng;
    rng.uniform(_nRndSeq, &numbers[0], engineImpl->getState(), 0, (size_t)(_nRndSeq - 1));
    float values[_maxSeqLength];
    for (uint32_t i = 0; i < _nRndSeq; i++)
    {
        values[i] = static_cast<float>(numbers[i]) / (_nRndSeq - 1);
    }
    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    _rndSeq        = context.allocate(par.type, _nRndSeq, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.copy(_rndSeq, 0, (void *)&values[0], _nRndSeq, 0, _nRndSeq, status);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

SelectIndexed * QuickSelectIndexed::create(Params & par, services::Status & status)
{
    QuickSelectIndexed * ret = new QuickSelectIndexed();
    if (!ret)
    {
        status |= Status(ErrorMemoryAllocationFailed);
        return nullptr;
    }
    status |= ret->init(par);
    if (!status.ok())
    {
        delete ret;
        return nullptr;
    }
    return ret;
}

services::Status DirectSelectIndexed::buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId, uint32_t nK)
{
    services::Status status;
    services::String fptypeName = getKeyFPType(vectorTypeId);
    auto buildOptions           = fptypeName;
    buildOptions.add("-cl-std=CL1.2 -D __K__=");
    char buffer[DAAL_MAX_STRING_SIZE];
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, nK);
    buildOptions.add(buffer);

    services::String cachekey("__daal_oneapi_internal_dselect_indexed_");
    cachekey.add(fptypeName);
    cachekey.add(buildOptions);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), direct_select_simd, buildOptions.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

SelectIndexed::Result & DirectSelectIndexed::selectIndices(const UniversalBuffer & dataVectors, uint32_t nK, uint32_t nVectors, uint32_t vectorSize,
                                                           uint32_t lastVectorSize, uint32_t vectorOffset, Result & result, services::Status & status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(QuickSelectIndexed.select);

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    status |= buildProgram(kernelFactory, dataVectors.type(), nK);
    if (!status.ok())
    {
        return result;
    }
    status |= runDirectSelectSimd(context, kernelFactory, dataVectors, nK, nVectors, vectorSize, lastVectorSize, vectorOffset, result);
    return result;
}

SelectIndexed * DirectSelectIndexed::create(Params & par, services::Status & status)
{
    DirectSelectIndexed * ret = new DirectSelectIndexed(par.nK);
    if (!ret)
    {
        status |= ErrorMemoryAllocationFailed;
        return nullptr;
    }
    return ret;
}

SelectIndexedFactory::SelectIndexedFactory()
{
    _entries << makeEntry<DirectSelectIndexed>();
    _entries << makeEntry<QuickSelectIndexed>();
}

SelectIndexed * SelectIndexedFactory::create(int nK, SelectIndexed::Params & par, services::Status & status)
{
    for (size_t i = 0; i < _entries.size(); i++)
    {
        if (_entries[i].inRange(nK))
        {
            return _entries[i].createMethod(par, status);
        }
    }
    status |= ErrorMethodNotImplemented;
    return nullptr;
}

} // namespace selection
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
