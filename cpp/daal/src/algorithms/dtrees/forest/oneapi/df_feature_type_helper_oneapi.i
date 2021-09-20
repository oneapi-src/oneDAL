/* file: df_feature_type_helper_oneapi.i */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
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
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    services::Status status;

    auto fptype_name    = getKeyFPType<algorithmFPType>();
    auto radixtype_name = getOpenCLKeyType<typename GetIntegerTypeForFPType<algorithmFPType>::Type>("radixIntType");
    auto build_options  = fptype_name + radixtype_name;

    services::String cachekey("__daal_algorithms_df_common_");
    cachekey.add(build_options);
    build_options.add(" -cl-std=CL1.2 ");

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), df_common_kernels, build_options.c_str(), status);

    return status;
}

template <typename algorithmFPType>
IndexedFeaturesOneAPI<algorithmFPType>::~IndexedFeaturesOneAPI()
{}

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
size_t IndexedFeaturesOneAPI<algorithmFPType>::getRequiredMemSize(size_t nCols, size_t nRows)
{
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nCols);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, sizeof(BinType), nRows * nCols);

    size_t requiredMem = sizeof(BinType) * (nCols + 1);

    requiredMem += sizeof(BinType) * nRows * nCols; // data vs ftrs bin map table (_fullData)
    return requiredMem;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::alloc(size_t nC, size_t nR)
{
    auto & context = services::internal::getDefaultContext();
    services::Status status;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nR, nC);
    _fullData = context.allocate(TypeIds::id<BinType>(), nR * nC, status);
    DAAL_CHECK_STATUS_VAR(status);

    _binOffsets = context.allocate(TypeIds::id<BinType>(), nC + 1, status);
    DAAL_CHECK_STATUS_VAR(status);

    _entries.reset(nC);
    DAAL_CHECK_MALLOC(_entries.get());
    _nCols     = nC;
    _nRows     = nR;
    _totalBins = 0;

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::extractColumn(const services::internal::Buffer<algorithmFPType> & data,
                                                                       UniversalBuffer & values, UniversalBuffer & indices, int32_t featureId,
                                                                       int32_t nFeatures, int32_t nRows)
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
        DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int32_t, nRows);

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
                                                                           UniversalBuffer & binBorders, int32_t nRows, int32_t maxBins)
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
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, int32_t, maxBins);
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
                                                                     UniversalBuffer & binBorders, UniversalBuffer & bins, int32_t nRows,
                                                                     int32_t nBins, int32_t maxBins, int32_t localSize, int32_t nLocalBlocks)
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
        DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int32_t, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binBorders, algorithmFPType, maxBins);
        DAAL_ASSERT_UNIVERSAL_BUFFER(bins, BinType, nRows);

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
                                                                     FeatureEntry & entry, int32_t nRows, const dtrees::internal::BinParams * pBinPrm)
{
    services::Status status;

    auto & context = services::internal::getDefaultContext();

    const int32_t maxBins      = pBinPrm->maxBins < nRows ? pBinPrm->maxBins : nRows;
    const int32_t localSize    = _preferableSubGroup;
    const int32_t nLocalBlocks = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);

    auto binOffsets = context.allocate(TypeIds::id<int32_t>(), maxBins, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto binBorders = context.allocate(TypeIds::id<algorithmFPType>(), maxBins, status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        auto binOffsetsHost = binOffsets.template get<int32_t>().toHost(ReadWriteMode::writeOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_MALLOC(binOffsetsHost.get());
        int32_t offset = 0;
        for (int32_t i = 0; i < maxBins; i++)
        {
            offset += (nRows + i) / maxBins;
            binOffsetsHost.get()[i] = offset - 1;
        }
    }

    DAAL_CHECK_STATUS_VAR(collectBinBorders(values, binOffsets, binBorders, nRows, maxBins));

    int32_t nBins = 0;
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(binBorders, algorithmFPType, maxBins);
        auto binBordersHost = binBorders.template get<algorithmFPType>().toHost(ReadWriteMode::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_MALLOC(binBordersHost.get());
        for (int32_t i = 0; i < maxBins; i++)
        {
            if (nBins == 0 || (nBins > 0 && binBordersHost.get()[i] != binBordersHost.get()[nBins - 1]))
            {
                binBordersHost.get()[nBins] = binBordersHost.get()[i];
                nBins++;
            }
        }
    }

    DAAL_CHECK_STATUS_VAR(computeBins(values, indices, binBorders, bins, nRows, nBins, maxBins, localSize, nLocalBlocks));

    entry.numIndices = static_cast<size_t>(nBins);
    entry.binBorders = binBorders;

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::makeIndex(const services::internal::Buffer<algorithmFPType> & data, int32_t featureId,
                                                                   int32_t nFeatures, int32_t nRows, const dtrees::internal::BinParams * pBinPrm,
                                                                   UniversalBuffer & _values, UniversalBuffer & _values_buf,
                                                                   UniversalBuffer & _indices, UniversalBuffer & _indices_buf, UniversalBuffer & bins,
                                                                   FeatureEntry & entry)
{
    DAAL_CHECK_STATUS_VAR(extractColumn(data, _values, _indices, featureId, nFeatures, nRows));
    DAAL_CHECK_STATUS_VAR(sort::RadixSort::sortIndices(_values, _indices, _values_buf, _indices_buf, nRows));
    DAAL_CHECK_STATUS_VAR(computeBins(_values, _indices, bins, entry, nRows, pBinPrm));
    return services::Status();
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::storeColumn(const UniversalBuffer & data, UniversalBuffer & fullData, int32_t featureId,
                                                                     int32_t nFeatures, int32_t nRows)
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
        DAAL_ASSERT_UNIVERSAL_BUFFER(data, BinType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(fullData, BinType, nRows * nFeatures);

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

    const size_t nRsz = nt.getNumberOfRows();
    const size_t nCsz = nt.getNumberOfColumns();

    if (nRsz > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    }
    if (nCsz > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    }

    const int32_t nC = static_cast<int32_t>(nCsz);
    const int32_t nR = static_cast<int32_t>(nRsz);

    services::Status status = alloc(nCsz, nRsz);
    DAAL_CHECK_STATUS_VAR(status);

    auto & context = services::internal::getDefaultContext();

    //allocating auxilliary buffers
    services::Collection<services::internal::sycl::UniversalBuffer> _data;

    DAAL_CHECK_MALLOC(_data.resize(nCsz));

    for (size_t i = 0; i < nCsz; i++)
    {
        _data[i] = context.allocate(TypeId::uint32, nRsz, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    auto _values = context.allocate(TypeIds::id<algorithmFPType>(), nRsz, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto _values_buf = context.allocate(TypeIds::id<algorithmFPType>(), nRsz, status);
    DAAL_CHECK_STATUS_VAR(status);

    auto _indices = context.allocate(TypeIds::id<int32_t>(), nRsz, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto _indices_buf = context.allocate(TypeIds::id<int32_t>(), nRsz, status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;

    if (nt.getDataLayout() == NumericTableIface::soa)
    {
        for (int32_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(nt.getBlockOfColumnValues(i, 0, nR, readOnly, dataBlock));
            auto dataBuffer = dataBlock.getBuffer();
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, 0, 1, nR, pBinPrm, _values, _values_buf, _indices, _indices_buf, _data[i], _entries[i]));
            DAAL_CHECK_STATUS_VAR(nt.releaseBlockOfColumnValues(dataBlock));
        }
    }
    else
    {
        DAAL_CHECK_STATUS_VAR(nt.getBlockOfRows(0, nR, readOnly, dataBlock));
        auto dataBuffer = dataBlock.getBuffer();
        for (int32_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, i, nC, nR, pBinPrm, _values, _values_buf, _indices, _indices_buf, _data[i], _entries[i]));
        }
        DAAL_CHECK_STATUS_VAR(nt.releaseBlockOfRows(dataBlock));
    }

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(_binOffsets, BinType, nC + 1);
        auto binOffsetsHost = _binOffsets.template get<BinType>().toHost(ReadWriteMode::writeOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_MALLOC(binOffsetsHost.get());
        BinType total = 0;
        for (int32_t i = 0; i < nC; i++)
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

} /* namespace internal */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */
