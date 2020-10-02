/* file: df_feature_type_helper_oneapi.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "src/externals/service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(df.common.oneapi);

using namespace daal::oneapi::internal;
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
static void buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    {
        auto fptype_name    = getKeyFPType<algorithmFPType>();
        auto radixtype_name = getOpenCLKeyType<typename GetIntegerTypeForFPType<algorithmFPType>::Type>("radixIntType");
        auto build_options  = fptype_name + radixtype_name;

        services::String cachekey("__daal_algorithms_df_common_");
        cachekey.add(build_options);
        build_options.add(" -cl-std=CL1.2 ");

        factory.build(ExecutionTargetIds::device, cachekey.c_str(), df_common_kernels, build_options.c_str());
    }
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
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    services::Status status;

    binBorders = context.allocate(TypeIds::id<algorithmFPType>(), numIndices, &status);
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::alloc(size_t nC, size_t nR)
{
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    services::Status status;

    DAAL_CHECK_MALLOC(_data.resize(nC));

    for (size_t i = 0; i < nC; i++)
    {
        _data[i] = context.allocate(TypeId::uint32, nR, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    _fullData = context.allocate(TypeId::uint32, nR * nC, &status);
    DAAL_CHECK_STATUS_VAR(status);

    _binOffsets = context.allocate(TypeId::uint32, nC + 1, &status);
    DAAL_CHECK_STATUS_VAR(status);

    _entries.reset(nC);
    DAAL_CHECK_MALLOC(_entries.get());
    _nCols     = nC;
    _nRows     = nR;
    _totalBins = 0;

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::extractColumn(const services::Buffer<algorithmFPType> & data, UniversalBuffer & values,
                                                                       UniversalBuffer & indices, int32_t featureId, int32_t nFeatures, int32_t nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.extractColumn);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    buildProgram<algorithmFPType>(factory);

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
services::Status IndexedFeaturesOneAPI<algorithmFPType>::collectBinBorders(UniversalBuffer & values, UniversalBuffer & binOffsets,
                                                                           UniversalBuffer & binBorders, int32_t nRows, int32_t maxBins)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.collectBinBorders);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    buildProgram<algorithmFPType>(factory);

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
services::Status IndexedFeaturesOneAPI<algorithmFPType>::computeBins(UniversalBuffer & values, UniversalBuffer & indices,
                                                                     UniversalBuffer & binBorders, UniversalBuffer & bins, int32_t nRows,
                                                                     int32_t nBins, int32_t localSize, int32_t nLocalBlocks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(indexedFeatures.computeBins);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    buildProgram<algorithmFPType>(factory);

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
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status IndexedFeaturesOneAPI<algorithmFPType>::computeBins(UniversalBuffer & values, UniversalBuffer & indices, UniversalBuffer & bins,
                                                                     FeatureEntry & entry, int32_t nRows, const dtrees::internal::BinParams * pBinPrm)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int32_t maxBins      = pBinPrm->maxBins < nRows ? pBinPrm->maxBins : nRows;
    const int32_t localSize    = _preferableSubGroup;
    const int32_t nLocalBlocks = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);

    auto binOffsets = context.allocate(TypeIds::id<int32_t>(), maxBins, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto binBorders = context.allocate(TypeIds::id<algorithmFPType>(), maxBins, &status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        auto binOffsetsHost = binOffsets.template get<int32_t>().toHost(ReadWriteMode::writeOnly);
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
        auto binBordersHost = binBorders.template get<algorithmFPType>().toHost(ReadWriteMode::readWrite);
        DAAL_CHECK_MALLOC(binBordersHost.get());
        for (int32_t i = 0; i < maxBins; i++)
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
services::Status IndexedFeaturesOneAPI<algorithmFPType>::makeIndex(const services::Buffer<algorithmFPType> & data, int32_t featureId,
                                                                   int32_t nFeatures, int32_t nRows, const dtrees::internal::BinParams * pBinPrm,
                                                                   UniversalBuffer & bins, FeatureEntry & entry)
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

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    buildProgram<algorithmFPType>(factory);

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

    services::Status status = alloc(nCsz, nRsz);
    DAAL_CHECK_STATUS_VAR(status);

    const int32_t nC = static_cast<int32_t>(nCsz);
    const int32_t nR = static_cast<int32_t>(nRsz);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    _values = context.allocate(TypeIds::id<algorithmFPType>(), nR, &status);
    DAAL_CHECK_STATUS_VAR(status);
    _values_buf = context.allocate(TypeIds::id<algorithmFPType>(), nR, &status);
    DAAL_CHECK_STATUS_VAR(status);

    _indices = context.allocate(TypeIds::id<int32_t>(), nR, &status);
    DAAL_CHECK_STATUS_VAR(status);
    _indices_buf = context.allocate(TypeIds::id<int32_t>(), nR, &status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;

    if (nt.getDataLayout() == NumericTableIface::soa)
    {
        for (int32_t i = 0; i < nC; i++)
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
        for (int32_t i = 0; i < nC; i++)
        {
            DAAL_CHECK_STATUS_VAR(makeIndex(dataBuffer, i, nC, nR, pBinPrm, _data[i], _entries[i]));
        }
        nt.releaseBlockOfRows(dataBlock);
    }

    {
        auto binOffsetsHost = _binOffsets.template get<int32_t>().toHost(ReadWriteMode::writeOnly);
        DAAL_CHECK_MALLOC(binOffsetsHost.get());
        size_t total = 0;
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
