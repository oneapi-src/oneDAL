/* file: gbt_feature_type_helper_oneapi.i */
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
//  GPU-dependent initialization of service data structure
//--
*/
#include "dtrees_feature_type_helper.h"

#include "service_data_utils.h"
#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(gbt.common.oneapi);

using namespace daal::oneapi::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{

template <typename FPType>
struct GetIntegerTypeForFPType;

template <>
struct GetIntegerTypeForFPType<float> {
    using Type = uint32_t;
};

template <>
struct GetIntegerTypeForFPType<double> {
    using Type = uint64_t;
};

template <typename IntType>
services::String getOpenCLKeyType(const services::String &typeName);

template <>
inline services::String getOpenCLKeyType<uint32_t>(const services::String &typeName) {
    return services::String(" -D ") + typeName + services::String("=uint ");
}

template <>
inline services::String getOpenCLKeyType<uint64_t>(const services::String &typeName) {
    return services::String(" -D ") + typeName + services::String("=ulong ");
}

template <typename algorithmFPType>
static void __buildProgram(ClKernelFactoryIface& factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    {
        auto fptype_name = getKeyFPType<algorithmFPType>();
        auto radixtype_name = getOpenCLKeyType<typename GetIntegerTypeForFPType<algorithmFPType>::Type>("radixIntType");
        auto build_options = fptype_name + radixtype_name;
        build_options.add("-cl-std=CL1.2");

        services::String cachekey("__daal_algorithms_gbt_common_");
        cachekey.add(fptype_name);
        cachekey.add(radixtype_name);
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), gbt_common_kernels, build_options.c_str());
    }
}

template <typename algorithmFPType>
IndexedFeaturesOneAPI<algorithmFPType>::~IndexedFeaturesOneAPI()
{
    delete [] _entries;
}

template <typename algorithmFPType>
IndexedFeaturesOneAPI<algorithmFPType>::FeatureEntry::~FeatureEntry()
{
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::FeatureEntry::allocBorders()
{
    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    services::Status status;

    binBorders = context.allocate(TypeIds::id<algorithmFPType>(), numIndices, &status);
    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::alloc(size_t nC, size_t nR)
{
    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    services::Status status;

    _data.resize(nC);

    for (size_t i = 0; i < nC; i++)
    {
        _data[i] = context.allocate(TypeId::uint32, nR, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    _fullData = context.allocate(TypeId::uint32, nR * nC, &status);
    DAAL_CHECK_STATUS_VAR(status);

    _binOffsets = context.allocate(TypeId::uint32, nC + 1, &status);
    DAAL_CHECK_STATUS_VAR(status);

    _entries = new FeatureEntry[nC];
    DAAL_CHECK_MALLOC(_entries);
    _nCols = nC;
    _nRows = nR;
    _totalBins = 0;
    return services::Status();
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::extractColumn(const services::Buffer<algorithmFPType>& data,
                                                     UniversalBuffer& values,
                                                     UniversalBuffer& indices,
                                                     int featureId,
                                                     int nFeatures,
                                                     int nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.extractColumn);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("extractColumn");

    {
        KernelArguments args(6);
        args.set(0, data, AccessModeIds::read);
        args.set(1, values, AccessModeIds::write);
        args.set(2, indices, AccessModeIds::write);
        args.set(3, featureId);
        args.set(4, nFeatures);
        args.set(5, nRows);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }
    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::radixScan(UniversalBuffer& values,
                                                 UniversalBuffer& partialHists,
                                                 int nRows,
                                                 int bitOffset,
                                                 int localSize,
                                                 int nLocalHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.radixScan);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("radixScan");

    {
        KernelArguments args(4);
        args.set(0, values, AccessModeIds::read);
        args.set(1, partialHists, AccessModeIds::write);
        args.set(2, nRows);
        args.set(3, bitOffset);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalHists);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::radixHistScan(UniversalBuffer& partialHists,
                                                     UniversalBuffer& partialPrefixHists,
                                                     int localSize,
                                                     int nSubgroupHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.radixHistScan);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("radixHistScan");

    {
        KernelArguments args(3);
        args.set(0, partialHists, AccessModeIds::read);
        args.set(1, partialPrefixHists, AccessModeIds::write);
        args.set(2, nSubgroupHists);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::radixReorder(UniversalBuffer& valuesSrc,
                                                    UniversalBuffer& indicesSrc,
                                                    UniversalBuffer& partialPrefixHists,
                                                    UniversalBuffer& valuesDst,
                                                    UniversalBuffer& indicesDst,
                                                    int nRows,
                                                    int bitOffset,
                                                    int localSize,
                                                    int nLocalHists)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.radixReorder);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("radixReorder");

    {
        KernelArguments args(7);
        args.set(0, valuesSrc, AccessModeIds::read);
        args.set(1, indicesSrc, AccessModeIds::read);
        args.set(2, partialPrefixHists, AccessModeIds::read);
        args.set(3, valuesDst, AccessModeIds::write);
        args.set(4, indicesDst, AccessModeIds::write);
        args.set(5, nRows);
        args.set(6, bitOffset);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalHists);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::radixSort(UniversalBuffer& values,
                                                 UniversalBuffer& indices,
                                                 UniversalBuffer& values_buf,
                                                 UniversalBuffer& indices_buf,
                                                 int nRows)
{
    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int radixBits = 4;
    const int subSize = _preferableSubGroup;
    const int localSize = _preferableSubGroup;
    const int nLocalHists = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);
    const int nSubgroupHists = nLocalHists * (localSize / subSize);

    auto partialHists = context.allocate(TypeIds::id<int>(), (nSubgroupHists + 1) << _radixBits, &status);
    auto partialPrefixHists = context.allocate(TypeIds::id<int>(), (nSubgroupHists + 1) << _radixBits, &status);

    DAAL_CHECK_STATUS_VAR(status);

    size_t rev = 0;

    for (size_t bitOffset = 0; bitOffset < 8 * sizeof(algorithmFPType); bitOffset += radixBits, rev ^= 1)
    {
        if (!rev)
        {
            DAAL_CHECK_STATUS_VAR(radixScan(values, partialHists, nRows, bitOffset, localSize, nLocalHists));
            DAAL_CHECK_STATUS_VAR(radixHistScan(partialHists, partialPrefixHists, localSize, nSubgroupHists));
            DAAL_CHECK_STATUS_VAR(radixReorder(values, indices, partialPrefixHists, values_buf, indices_buf, nRows, bitOffset, localSize, nLocalHists));
        }
        else
        {
            DAAL_CHECK_STATUS_VAR(radixScan(values_buf, partialHists, nRows, bitOffset, localSize, nLocalHists));
            DAAL_CHECK_STATUS_VAR(radixHistScan(partialHists, partialPrefixHists, localSize, nSubgroupHists));
            DAAL_CHECK_STATUS_VAR(radixReorder(values_buf, indices_buf, partialPrefixHists, values, indices, nRows, bitOffset, localSize, nLocalHists));
        }
    }

    DAAL_ASSERT(rev == 0); // if not, we need to swap values/indices and values_buf/indices_buf

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::collectBinBorders(UniversalBuffer& values,
                                                    UniversalBuffer& binOffsets,
                                                    UniversalBuffer& binBorders,
                                                    int nRows,
                                                    int maxBins)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.collectBinBorders);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("collectBinBorders");

    {
        KernelArguments args(3);
        args.set(0, values, AccessModeIds::read);
        args.set(1, binOffsets, AccessModeIds::read);
        args.set(2, binBorders, AccessModeIds::write);

        KernelRange global_range(maxBins);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::computeBins(UniversalBuffer& values,
                                                   UniversalBuffer& indices,
                                                   UniversalBuffer& binBorders,
                                                   UniversalBuffer& bins,
                                                   int nRows,
                                                   int nBins,
                                                   int localSize,
                                                   int nLocalBlocks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.computeBins);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("computeBins");

    {
        KernelArguments args(6);
        args.set(0, values, AccessModeIds::read);
        args.set(1, indices, AccessModeIds::read);
        args.set(2, binBorders, AccessModeIds::read);
        args.set(3, bins, AccessModeIds::write);
        args.set(4, nRows);
        args.set(5, nBins);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalBlocks);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::computeBins(UniversalBuffer& values,
                                                                     UniversalBuffer& indices,
                                                                     UniversalBuffer& bins,
                                                                     FeatureEntry& entry,
                                                                     int nRows,
                                                                     const dtrees::internal::BinParams* pBinPrm)
{
    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int maxBins = pBinPrm->maxBins < nRows ? pBinPrm->maxBins : nRows;
    const int localSize = _preferableSubGroup;
    const int nLocalBlocks = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);

    auto binOffsets = context.allocate(TypeIds::id<int>(), maxBins, &status);
    auto binBorders = context.allocate(TypeIds::id<algorithmFPType>(), maxBins, &status);

    DAAL_CHECK_STATUS_VAR(status);

    {
        auto binOffsetsHost = binOffsets.template get<int>().toHost(ReadWriteMode::writeOnly);
        int offset = 0;
        for (int i = 0; i < maxBins; i++)
        {
            offset += (nRows + i) / maxBins;
            binOffsetsHost.get()[i] = offset - 1;
        }
    }

    DAAL_CHECK_STATUS_VAR(collectBinBorders(values, binOffsets, binBorders, nRows, maxBins));

    int nBins = 0;
    {
        auto binBordersHost = binBorders.template get<algorithmFPType>().toHost(ReadWriteMode::readWrite);
        for (int i = 0; i < maxBins; i++)
        {
            if (nBins == 0 || binBordersHost.get()[i] != binBordersHost.get()[nBins - 1])
            {
                binBordersHost.get()[nBins] = binBordersHost.get()[i];
                nBins++;
            }
        }
    }

    DAAL_CHECK_STATUS_VAR(computeBins(values, indices, binBorders, bins, nRows, nBins, localSize, nLocalBlocks));

    entry.numIndices = nBins;
    entry.binBorders = binBorders;

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::makeIndex(const services::Buffer<algorithmFPType>& data,
                                                                   int featureId,
                                                                   int nFeatures,
                                                                   int nRows,
                                                                   const dtrees::internal::BinParams* pBinPrm,
                                                                   UniversalBuffer& bins,
                                                                   FeatureEntry& entry)
{
    DAAL_CHECK_STATUS_VAR(extractColumn(data, _values, _indices, featureId, nFeatures, nRows));
    DAAL_CHECK_STATUS_VAR(radixSort(_values, _indices, _values_buf, _indices_buf, nRows));
    DAAL_CHECK_STATUS_VAR(computeBins(_values, _indices, bins, entry, nRows, pBinPrm));
    return services::Status();
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::storeColumn(const UniversalBuffer& data,
                                                                     UniversalBuffer& fullData,
                                                                     int featureId,
                                                                     int nFeatures,
                                                                     int nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.storeColumn);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto& factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("storeColumn");

    {
        KernelArguments args(5);
        args.set(0, data, AccessModeIds::read);
        args.set(1, fullData, AccessModeIds::write);
        args.set(2, featureId);
        args.set(3, nFeatures);
        args.set(4, nRows);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::init(NumericTable& nt, const dtrees::internal::FeatureTypes* featureTypes,
    const dtrees::internal::BinParams* pBinPrm)
{
    dtrees::internal::FeatureTypes autoFT;
    if(!featureTypes)
    {
        DAAL_CHECK_MALLOC(autoFT.init(nt));
        featureTypes = &autoFT;
    }

    const size_t nC = nt.getNumberOfColumns();
    const size_t nR = nt.getNumberOfRows();

    _maxNumIndices = 0;
    services::Status status = alloc(nC, nR);
    if(!status)
        return status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    _values = context.allocate(TypeIds::id<algorithmFPType>(), nR, &status);
    _values_buf = context.allocate(TypeIds::id<algorithmFPType>(), nR, &status);

    _indices = context.allocate(TypeIds::id<int>(), nR, &status);
    _indices_buf = context.allocate(TypeIds::id<int>(), nR, &status);

    BlockDescriptor<algorithmFPType> dataBlock;

    if (nt.getDataLayout() == NumericTableIface::soa)
    {
        for (size_t i = 0; i < nC; i++)
        {
            nt.getBlockOfColumnValues(i, 0, nR, readOnly, dataBlock);
            auto dataBuffer = dataBlock.getBuffer();
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, 0, 1, nR, pBinPrm, _data[i], _entries[i]));
            nt.releaseBlockOfColumnValues(dataBlock);
        }
    }
    else
    {
        nt.getBlockOfRows(0, nR, readOnly, dataBlock);
        auto dataBuffer = dataBlock.getBuffer();
        for (size_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, i, nC, nR, pBinPrm, _data[i], _entries[i]));
        }
        nt.releaseBlockOfRows(dataBlock);
    }

    {
        auto binOffsetsHost = _binOffsets.template get<int>().toHost(ReadWriteMode::writeOnly);
        size_t total = 0;
        for (size_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(storeColumn(_data[i], _fullData, i, nC, nR));
            binOffsetsHost.get()[i] = total;
            _entries[i].offset = total;
            total += _entries[i].numIndices;
        }
        binOffsetsHost.get()[nC] = total;
        _totalBins = total;
    }

    return status;
}

template<typename algorithmFPType>
services::Status TreeNodeStorage::allocate(const gbt::internal::IndexedFeaturesOneAPI<algorithmFPType>& indexedFeatures)
{
    services::Status status;
    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    _histogramsForFeatures = context.allocate(TypeIds::id<algorithmFPType>(), indexedFeatures.totalBins() * 2, &status);

    return status;
}

template<typename algorithmFPType>
BestSplitOneAPI<algorithmFPType>::BestSplitOneAPI() : _impurityDecrease(-services::internal::MaxVal<algorithmFPType>::get()), _featureIndex(-1), _featureValue(0),
                                                      _leftGTotal(0.0), _leftHTotal(0.0), _rightGTotal(0.0), _rightHTotal(0.0) { }

} /* namespace internal */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */
