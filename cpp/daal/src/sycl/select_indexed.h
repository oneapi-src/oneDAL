/* file: select_indexed.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include "algorithms/engines/engine.h"
#include "services/internal/buffer.h"
#include "services/daal_defines.h"
#include "services/internal/error_handling_helpers.h"
#include "services/internal/sycl/types_utils.h"
#include "services/internal/sycl/execution_context.h"

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
class SelectIndexed
{
public:
    struct Result
    {
        UniversalBuffer values;
        UniversalBuffer indices;

        Result() {}
        Result(ExecutionContextIface & context, uint32_t nK, uint32_t nVectors, TypeId valueType, services::Status & status)
            : values(context.allocate(valueType, nVectors * nK, status)), indices(context.allocate(TypeIds::id<int>(), nVectors * nK, status))
        {}
    };
    struct Params
    {
        Params(uint32_t nK, TypeId fptype, uint32_t size, daal::algorithms::engines::EnginePtr eng)
            : nK(nK), type(fptype), dataSize(size), engine(eng)
        {}
        uint32_t nK;
        TypeId type;
        uint32_t dataSize;
        daal::algorithms::engines::EnginePtr engine;
    };

public:
    virtual ~SelectIndexed() {}
    virtual Result & selectIndices(const UniversalBuffer & dataVectors, uint32_t nK, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                                   uint32_t vectorOffset, Result & result, services::Status & status) = 0;
    static services::Status convertIndicesToLabels(const UniversalBuffer & indices, const UniversalBuffer & labels, uint32_t nVectors,
                                                   uint32_t vectorSize, uint32_t vectorOffset);
    services::Status selectNearestDistancesAndLabels(const UniversalBuffer & distances, const UniversalBuffer & dataLabels, uint32_t nK,
                                                     uint32_t nVectors, uint32_t vectorSize, uint32_t vectorOffset, uint32_t labelOffset,
                                                     Result & result)
    {
        services::Status status;
        selectIndices(distances, nK, nVectors, vectorSize, vectorSize, vectorOffset, result, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_STATUS_VAR(convertIndicesToLabels(result.indices, dataLabels, nVectors, nK, labelOffset));
        return services::Status();
    }
};

class SelectIndexedFactory
{
public:
    SelectIndexedFactory();
    SelectIndexed * create(int nK, SelectIndexed::Params & par, services::Status & st);

private:
    typedef SelectIndexed * (*CreateFuncType)(SelectIndexed::Params & par, services::Status & st);
    struct Entry
    {
        int minK;
        int maxK;
        CreateFuncType createMethod;
        bool inRange(int nK) const { return nK >= minK && nK <= maxK; }
    };
    template <class T>
    Entry makeEntry()
    {
        Entry e;
        e.minK         = T::minK;
        e.maxK         = T::maxK;
        e.createMethod = T::create;
        return e;
    }
    daal::services::Collection<Entry> _entries;
};

class QuickSelectIndexed : public SelectIndexed
{
public:
    static const int minK = 33;
    static const int maxK = INT_MAX;
    static SelectIndexed * create(Params & par, daal::services::Status & st);
    virtual Result & selectIndices(const UniversalBuffer & dataVectors, uint32_t nK, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                                   uint32_t vectorOffset, Result & result, services::Status & status)
    {
        status |= adjustIndexBuffer(nVectors, vectorSize);
        DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);
        return selectIndices(dataVectors, _indices, _rndSeq, _nRndSeq, nK, nVectors, vectorSize, lastVectorSize, vectorOffset, result, status);
    }

private:
    QuickSelectIndexed() {}
    services::Status adjustIndexBuffer(uint32_t number, uint32_t size);
    static Result & selectIndices(const UniversalBuffer & dataVectors, const UniversalBuffer & tempIndices, const UniversalBuffer & rndSeq,
                                  uint32_t nRndSeq, uint32_t nK, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                                  uint32_t vectorOffset, Result & result, services::Status & status);
    daal::services::Status init(Params & par);
    static services::Status buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId);

private:
    static const uint32_t _maxSeqLength = 1024;
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
    static SelectIndexed * create(Params & par, daal::services::Status & st);
    virtual Result & selectIndices(const UniversalBuffer & dataVectors, uint32_t nK, uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize,
                                   uint32_t vectorOffset, Result & result, services::Status & status);

private:
    static services::Status buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId, uint32_t nK);

private:
    DirectSelectIndexed(uint32_t nK) : _nK(nK) {}
    uint32_t _nK;
};

} // namespace selection
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
