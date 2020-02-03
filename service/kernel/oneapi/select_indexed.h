/* file: select_indexed.h */
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

#ifndef __SELECT_INDEXED_H__
#define __SELECT_INDEXED_H__

#include "service/kernel/oneapi/math_service_types.h"
#include "services/buffer.h"
#include "service/kernel/oneapi/cl_kernels/select_indexed.cl"
#include "oneapi/internal/types_utils.h"
#include "oneapi/internal/execution_context.h"

#include "cl_kernels/select_indexed.cl"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace selection
{
class SelectIndexed
{
public:
    struct Result
    {
        UniversalBuffer values;
        UniversalBuffer indices;

        Result() {}
        Result(ExecutionContextIface & context, uint32_t K, uint32_t nVectors, TypeId valueType, services::Status * status)
            : values(context.allocate(valueType, nVectors * K, status)), indices(context.allocate(TypeIds::id<int>(), nVectors * K, status))
        {}
    };
    struct Params
    {
        Params(uint32_t K, TypeId fptype, uint32_t size, daal::algorithms::engines::EnginePtr eng) : nK(K), type(fptype), dataSize(size), engine(eng)
        {}
        uint32_t nK;
        TypeId type;
        uint32_t dataSize;
        daal::algorithms::engines::EnginePtr engine;
    };

public:
    virtual ~SelectIndexed() {}
    virtual Result select(const UniversalBuffer & dataVectors, uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                          uint32_t vectorOffset, services::Status * status)                    = 0;
    virtual Result & select(const UniversalBuffer & dataVectors, uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                            uint32_t vectorOffset, Result & result, services::Status * status) = 0;
    static void convert(const UniversalBuffer & indices, const UniversalBuffer & labels, uint32_t nVectors, uint32_t vectorSize,
                        uint32_t vectorOffset, services::Status * status);
    void selectLabels(const UniversalBuffer & distances, const UniversalBuffer & dataLabels, uint32_t nK, uint32_t nVectors, uint32_t vectorSize,
                      uint32_t vectorOffset, uint32_t labelOffset, Result & result, services::Status * status)
    {
        select(distances, nK, nVectors, vectorSize, vectorSize, vectorOffset, result, status);
        DAAL_CHECK_STATUS_PTR(status);
        convert(result.indices, dataLabels, nVectors, nK, labelOffset, status);
    }
    void selectLabels(const UniversalBuffer & distances, const UniversalBuffer & dataLabels, uint32_t nK, uint32_t nVectors, uint32_t vectorSize,
                      uint32_t lastVectorSize, uint32_t vectorOffset, uint32_t labelOffset, Result & result, services::Status * status)
    {
        select(distances, nK, nVectors, vectorSize, lastVectorSize, vectorOffset, result, status);
        DAAL_CHECK_STATUS_PTR(status);
        convert(result.indices, dataLabels, nVectors, nK, labelOffset, status);
    }
};

class SelectIndexedFactory
{
public:
    SelectIndexedFactory();
    SelectIndexed * Create(int K, SelectIndexed::Params & par, daal::services::Status * st);

private:
    typedef SelectIndexed * (*CreateFuncType)(SelectIndexed::Params & par, daal::services::Status * st);
    struct Entry
    {
        int minK;
        int maxK;
        CreateFuncType createMethod;
        bool inRange(int K) { return K >= minK && K <= maxK; }
    };
    template <class T>
    Entry makeEntry()
    {
        Entry e;
        e.minK         = T::minK;
        e.maxK         = T::maxK;
        e.createMethod = T::Create;
        return e;
    }
    daal::services::Collection<Entry> _entries;
};

class QuickSelectIndexed : public SelectIndexed
{
public:
    static const int minK = 33;
    static const int maxK = INT_MAX;
    static SelectIndexed * Create(Params & par, daal::services::Status * st);
    virtual Result select(const UniversalBuffer & dataVectors, uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                          uint32_t vectorOffset, services::Status * status)
    {
        adjustIndexBuffer(nVectors, vectorSize, status);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, Result());
        return select(dataVectors, _indices, _rndSeq, _nRndSeq, K, nVectors, vectorSize, lastVectorSize, vectorOffset, status);
    }
    virtual Result & select(const UniversalBuffer & dataVectors, uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                            uint32_t vectorOffset, Result & result, services::Status * status)
    {
        adjustIndexBuffer(nVectors, vectorSize, status);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);
        return select(dataVectors, _indices, _rndSeq, _nRndSeq, K, nVectors, vectorSize, lastVectorSize, vectorOffset, result, status);
    }

private:
    QuickSelectIndexed() {}
    void adjustIndexBuffer(uint32_t number, uint32_t size, services::Status * status);
    static Result select(const UniversalBuffer & dataVectors, const UniversalBuffer & tempIndices, const UniversalBuffer & rndSeq, uint32_t nRndSeq,
                         uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset,
                         services::Status * status);
    static Result & select(const UniversalBuffer & dataVectors, const UniversalBuffer & tempIndices, const UniversalBuffer & rndSeq, uint32_t nRndSeq,
                           uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset, Result & result,
                           services::Status * status);
    daal::services::Status init(Params & par);
    static void buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId, services::Status * status);

private:
    UniversalBuffer _indices;
    uint32_t _indexSize = 0;
    UniversalBuffer _rndSeq;
    uint32_t _nRndSeq = 0;
};

class DirectSelectIndexed : public SelectIndexed
{
public:
    static const int minK = 1;
    static const int maxK = 32;
    static SelectIndexed * Create(Params & par, daal::services::Status * st);
    virtual Result select(const UniversalBuffer & dataVectors, uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                          uint32_t vectorOffset, services::Status * status);
    virtual Result & select(const UniversalBuffer & dataVectors, uint32_t K, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                            uint32_t vectorOffset, Result & result, services::Status * status);

private:
    static void buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId, uint32_t K, services::Status * status);

private:
    DirectSelectIndexed(uint32_t K) : _K(K) {}
    uint32_t _K;
};

} // namespace selection
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
