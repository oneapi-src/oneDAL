/* file: gbt_feature_type_helper_oneapi.i */
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
//  GPU-dependent initialization of service data structure
//--
*/
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"

#include "src/services/service_data_utils.h"
#include "src/sycl/sorter.h"
#include "src/externals/service_profiler.h"

using namespace daal::services::internal::sycl;

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
struct GetIntegerTypeForFPType<float>
{
    using Type = uint32_t;
};

template <>
struct GetIntegerTypeForFPType<double>
{
    using Type = uint64_t;
};

template <typename IntType>
services::String getOpenCLKeyType(const services::String & typeName);

template <>
inline services::String getOpenCLKeyType<uint32_t>(const services::String & typeName)
{
    return services::String(" -D ") + typeName + services::String("=uint ");
}

template <>
inline services::String getOpenCLKeyType<uint64_t>(const services::String & typeName)
{
    return services::String(" -D ") + typeName + services::String("=ulong ");
}

template <typename algorithmFPType>
static services::Status buildProgram(ClKernelFactoryIface & factory)
{
    services::Status status;

    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    auto fptype_name    = getKeyFPType<algorithmFPType>();
    auto radixtype_name = getOpenCLKeyType<typename GetIntegerTypeForFPType<algorithmFPType>::Type>("radixIntType");
    auto build_options  = fptype_name + radixtype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_gbt_common_");
    cachekey.add(fptype_name);
    cachekey.add(radixtype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), gbt_common_kernels, build_options.c_str(), status);

    return status;
}

template <typename algorithmFPType>
IndexedFeaturesOneAPI<algorithmFPType>::~IndexedFeaturesOneAPI()
{
    delete[] _entries;
    _entries = nullptr;
}

template <typename algorithmFPType>
IndexedFeaturesOneAPI<algorithmFPType>::FeatureEntry::~FeatureEntry()
{}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::FeatureEntry::allocBorders()
{
    auto & context = services::internal::getDefaultContext();
    services::Status status;

    binBorders = context.allocate(TypeIds::id<algorithmFPType>(), numIndices, status);
    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::alloc(uint32_t nC, uint32_t nR)
{
    auto & context = services::internal::getDefaultContext();
    services::Status status;

    if (!_data.resize(nC))
    {
        return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
    }

    for (uint32_t i = 0; i < nC; i++)
    {
        _data[i] = context.allocate(TypeId::uint32, nR, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nR, nC);
    _fullData = context.allocate(TypeId::uint32, nR * nC, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_OVERFLOW_CHECK_BY_ADDING(uint32_t, nC, 1);
    _binOffsets = context.allocate(TypeId::uint32, nC + 1, status);
    DAAL_CHECK_STATUS_VAR(status);

    _entries = new FeatureEntry[nC];
    DAAL_CHECK_MALLOC(_entries);
    _nCols     = nC;
    _nRows     = nR;
    _totalBins = 0;
    return services::Status();
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::extractColumn(const services::internal::Buffer<algorithmFPType> & data,
                                                                       UniversalBuffer & values, UniversalBuffer & indices, uint32_t featureId,
                                                                       uint32_t nFeatures, uint32_t nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.extractColumn);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram<algorithmFPType>(factory));

    auto kernel = factory.getKernel("extractColumn", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(data), algorithmFPType, nRows * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(values, algorithmFPType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int, nRows);

        KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, values, AccessModeIds::write);
        args.set(2, indices, AccessModeIds::write);
        args.set(3, featureId);
        args.set(4, nFeatures);
        args.set(5, nRows);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }
    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::collectBinBorders(UniversalBuffer & values, UniversalBuffer & binOffsets,
                                                                           UniversalBuffer & binBorders, uint32_t nRows, uint32_t maxBins)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.collectBinBorders);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    status |= buildProgram<algorithmFPType>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("collectBinBorders", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(values, algorithmFPType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, int, maxBins);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binBorders, algorithmFPType, maxBins);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, values, AccessModeIds::read);
        args.set(1, binOffsets, AccessModeIds::read);
        args.set(2, binBorders, AccessModeIds::write);

        KernelRange global_range(maxBins);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::computeBins(UniversalBuffer & values, UniversalBuffer & indices,
                                                                     UniversalBuffer & binBorders, UniversalBuffer & bins, uint32_t nRows,
                                                                     uint32_t nBins, uint32_t maxBins, uint32_t localSize, uint32_t nLocalBlocks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.computeBins);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    status |= buildProgram<algorithmFPType>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("computeBins", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(values, algorithmFPType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binBorders, algorithmFPType, maxBins);
        DAAL_ASSERT_UNIVERSAL_BUFFER(bins, uint32_t, nRows);

        KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, values, AccessModeIds::read);
        args.set(1, indices, AccessModeIds::read);
        args.set(2, binBorders, AccessModeIds::read);
        args.set(3, bins, AccessModeIds::write);
        args.set(4, nRows);
        args.set(5, nBins);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalBlocks);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::computeBins(UniversalBuffer & values, UniversalBuffer & indices, UniversalBuffer & bins,
                                                                     FeatureEntry & entry, uint32_t nRows,
                                                                     const dtrees::internal::BinParams * pBinPrm)
{
    services::Status status;

    auto & context = services::internal::getDefaultContext();

    const uint32_t maxBins      = pBinPrm->maxBins < nRows ? pBinPrm->maxBins : nRows;
    const uint32_t localSize    = _preferableSubGroup;
    const uint32_t nLocalBlocks = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);

    auto binOffsets = context.allocate(TypeIds::id<int>(), maxBins, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto binBorders = context.allocate(TypeIds::id<algorithmFPType>(), maxBins, status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, int, maxBins);
        auto binOffsetsHost = binOffsets.template get<int>().toHost(ReadWriteMode::writeOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        int offset = 0;
        for (int i = 0; i < maxBins; i++)
        {
            offset += (nRows + i) / maxBins;
            binOffsetsHost.get()[i] = offset - 1;
        }
    }

    DAAL_CHECK_STATUS_VAR(collectBinBorders(values, binOffsets, binBorders, nRows, maxBins));

    uint32_t nBins = 0;
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(binBorders, algorithmFPType, maxBins);
        auto binBordersHost = binBorders.template get<algorithmFPType>().toHost(ReadWriteMode::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);
        for (uint32_t i = 0; i < maxBins; i++)
        {
            if (nBins == 0 || (nBins > 0 && binBordersHost.get()[i] != binBordersHost.get()[nBins - 1]))
            {
                binBordersHost.get()[nBins] = binBordersHost.get()[i];
                nBins++;
            }
        }
    }

    DAAL_CHECK_STATUS_VAR(computeBins(values, indices, binBorders, bins, nRows, nBins, maxBins, localSize, nLocalBlocks));

    entry.numIndices = nBins;
    entry.binBorders = binBorders;

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::makeIndex(const services::internal::Buffer<algorithmFPType> & data, uint32_t featureId,
                                                                   uint32_t nFeatures, uint32_t nRows, const dtrees::internal::BinParams * pBinPrm,
                                                                   UniversalBuffer & bins, FeatureEntry & entry)
{
    DAAL_CHECK_STATUS_VAR(extractColumn(data, _values, _indices, featureId, nFeatures, nRows));
    DAAL_CHECK_STATUS_VAR(sort::RadixSort::sortIndices(_values, _indices, _values_buf, _indices_buf, nRows));
    DAAL_CHECK_STATUS_VAR(computeBins(_values, _indices, bins, entry, nRows, pBinPrm));
    return services::Status();
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::storeColumn(const UniversalBuffer & data, UniversalBuffer & fullData, uint32_t featureId,
                                                                     uint32_t nFeatures, uint32_t nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.storeColumn);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    status |= buildProgram<algorithmFPType>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("storeColumn", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(fullData, uint32_t, nRows * nFeatures);

        KernelArguments args(5, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, fullData, AccessModeIds::write);
        args.set(2, featureId);
        args.set(3, nFeatures);
        args.set(4, nRows);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::init(NumericTable & nt, const dtrees::internal::FeatureTypes * featureTypes,
                                                              const dtrees::internal::BinParams * pBinPrm)
{
    dtrees::internal::FeatureTypes autoFT;
    if (!featureTypes)
    {
        DAAL_CHECK_MALLOC(autoFT.init(nt));
        featureTypes = &autoFT;
    }

    if (nt.getNumberOfRows() > static_cast<size_t>(UINT_MAX) || nt.getNumberOfColumns() > static_cast<size_t>(UINT_MAX))
    {
        return Status(ErrorBufferSizeIntegerOverflow);
    }

    const uint32_t nC = static_cast<uint32_t>(nt.getNumberOfColumns());
    const uint32_t nR = static_cast<uint32_t>(nt.getNumberOfRows());

    _maxNumIndices          = 0;
    services::Status status = alloc(nC, nR);
    DAAL_CHECK_STATUS_VAR(status);

    auto & context = services::internal::getDefaultContext();

    _values = context.allocate(TypeIds::id<algorithmFPType>(), nR, status);
    DAAL_CHECK_STATUS_VAR(status);
    _values_buf = context.allocate(TypeIds::id<algorithmFPType>(), nR, status);
    DAAL_CHECK_STATUS_VAR(status);

    _indices = context.allocate(TypeIds::id<int>(), nR, status);
    DAAL_CHECK_STATUS_VAR(status);
    _indices_buf = context.allocate(TypeIds::id<int>(), nR, status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;

    if (nt.getDataLayout() == NumericTableIface::soa)
    {
        for (uint32_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(nt.getBlockOfColumnValues(i, 0, nR, readOnly, dataBlock));
            auto dataBuffer = dataBlock.getBuffer();
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, 0, 1, nR, pBinPrm, _data[i], _entries[i]));
            DAAL_CHECK_STATUS_VAR(nt.releaseBlockOfColumnValues(dataBlock));
        }
    }
    else
    {
        DAAL_CHECK_STATUS_VAR(nt.getBlockOfRows(0, nR, readOnly, dataBlock));
        auto dataBuffer = dataBlock.getBuffer();
        for (uint32_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, i, nC, nR, pBinPrm, _data[i], _entries[i]));
        }
        DAAL_CHECK_STATUS_VAR(nt.releaseBlockOfRows(dataBlock));
    }

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(_binOffsets, uint32_t, nC + 1);
        auto binOffsetsHost = _binOffsets.template get<int>().toHost(ReadWriteMode::writeOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        int total = 0;
        for (uint32_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(storeColumn(_data[i], _fullData, i, nC, nR));
            binOffsetsHost.get()[i] = total;
            _entries[i].offset      = total;
            total += _entries[i].numIndices;
        }
        binOffsetsHost.get()[nC] = total;
        _totalBins               = total;
    }

    return status;
}

template <typename algorithmFPType>
services::Status TreeNodeStorage::allocate(const gbt::internal::IndexedFeaturesOneAPI<algorithmFPType> & indexedFeatures)
{
    services::Status status;
    auto & context = services::internal::getDefaultContext();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, indexedFeatures.totalBins(), 2);
    _histogramsForFeatures = context.allocate(TypeIds::id<algorithmFPType>(), indexedFeatures.totalBins() * 2, status);

    return status;
}

template <typename algorithmFPType>
BestSplitOneAPI<algorithmFPType>::BestSplitOneAPI()
    : _impurityDecrease(-services::internal::MaxVal<algorithmFPType>::get()),
      _featureIndex(-1),
      _featureValue(0),
      _leftGTotal(0.0),
      _leftHTotal(0.0),
      _rightGTotal(0.0),
      _rightHTotal(0.0)
{}

} /* namespace internal */
} // namespace gbt
} /* namespace algorithms */
} /* namespace daal */
