/* file: df_feature_type_helper_oneapi.h */
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
//  Implementation of a service class that provides optimal access to the feature types
//--
*/

#ifndef __DF_FEATURE_TYPE_HELPER_ONEAPI_H__
#define __DF_FEATURE_TYPE_HELPER_ONEAPI_H__

#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/forest/oneapi/df_feature_type_helper_oneapi.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/dtrees/service_array.h"
#include "src/services/service_arrays.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/data_management/service_numeric_table.h"

#include "src/algorithms/dtrees/forest/oneapi/cl_kernels/df_common_kernels.cl"

#include "sycl/internal/execution_context.h"
#include "sycl/internal/types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// IndexedFeatures. Creates and stores index of every feature
// Sorts every feature and creates the mapping: features value -> index of the value
// in the sorted array of unique values of the feature in increasing order
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType>
class IndexedFeaturesOneAPI
{
public:
    typedef size_t IndexType;

    struct FeatureEntry
    {
        DAAL_NEW_DELETE();
        IndexType numIndices = 0; //number of indices or bins
        IndexType offset     = 0;
        oneapi::internal::UniversalBuffer binBorders; //right bin borders

        services::Status allocBorders();
        ~FeatureEntry();
    };

public:
    //IndexedFeaturesOneAPI() : _data(), _sizeOfIndex(sizeof(IndexType)), _nCols(0), _nRows(0) {}
    IndexedFeaturesOneAPI() : _data(), _nCols(0), _nRows(0) {}
    ~IndexedFeaturesOneAPI();

    services::Status init(NumericTable & nt, const dtrees::internal::FeatureTypes * featureTypes, const dtrees::internal::BinParams * pBinPrm);

    //get max number of indices for that feature
    IndexType numIndices(size_t iCol) const { return _entries[iCol].numIndices; }

    IndexType totalBins() const { return _totalBins; }

    oneapi::internal::UniversalBuffer & binBorders(size_t iCol) { return _entries[iCol].binBorders; }

    oneapi::internal::UniversalBuffer & binOffsets() { return _binOffsets; }

    oneapi::internal::UniversalBuffer & getFullData() { return _fullData; }

    //for low-level optimization
    const oneapi::internal::UniversalBuffer & getFeature(size_t iFeature) const { return _data[iFeature]; }

    size_t nRows() const { return _nRows; }
    size_t nCols() const { return _nCols; }

protected:
    services::Status alloc(size_t nCols, size_t nRows);

    services::Status extractColumn(const services::Buffer<algorithmFPType> & data, oneapi::internal::UniversalBuffer & values,
                                   oneapi::internal::UniversalBuffer & indices, int32_t featureId, int32_t nFeatures, int32_t nRows);

    services::Status collectBinBorders(oneapi::internal::UniversalBuffer & values, oneapi::internal::UniversalBuffer & binOffsets,
                                       oneapi::internal::UniversalBuffer & binBorders, int32_t nRows, int32_t maxBins);

    services::Status computeBins(oneapi::internal::UniversalBuffer & values, oneapi::internal::UniversalBuffer & indices,
                                 oneapi::internal::UniversalBuffer & binBorders, oneapi::internal::UniversalBuffer & bins, int32_t nRows,
                                 int32_t nBins, int32_t localSize, int32_t nLocalBlocks);

    services::Status computeBins(oneapi::internal::UniversalBuffer & values, oneapi::internal::UniversalBuffer & indices,
                                 oneapi::internal::UniversalBuffer & bins, FeatureEntry & entry, int32_t nRows,
                                 const dtrees::internal::BinParams * pBinPrm);

    services::Status makeIndex(const services::Buffer<algorithmFPType> & data, int32_t featureId, int32_t nFeatures, int32_t nRows,
                               const dtrees::internal::BinParams * pBinPrm, oneapi::internal::UniversalBuffer & bins, FeatureEntry & entry);

    services::Status storeColumn(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & fullData, int32_t featureId,
                                 int32_t nFeatures, int32_t nRows);

protected:
    services::Collection<oneapi::internal::UniversalBuffer> _data;
    oneapi::internal::UniversalBuffer _fullData;
    oneapi::internal::UniversalBuffer _binOffsets;
    daal::internal::TArray<FeatureEntry, sse2> _entries;
    size_t _nRows;
    size_t _nCols;
    IndexType _totalBins;

    oneapi::internal::UniversalBuffer _values;
    oneapi::internal::UniversalBuffer _values_buf;
    oneapi::internal::UniversalBuffer _indices;
    oneapi::internal::UniversalBuffer _indices_buf;

    static constexpr size_t _int32max = static_cast<size_t>(services::internal::MaxVal<int32_t>::get());

    const int32_t _preferableSubGroup = 16; // preferable maximal sub-group size
};

} /* namespace internal */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
