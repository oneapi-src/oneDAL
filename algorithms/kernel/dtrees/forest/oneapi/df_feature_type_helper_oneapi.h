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

#include "algorithms/kernel/dtrees/dtrees_feature_type_helper.h"
#include "algorithms/kernel/dtrees/forest/oneapi/df_feature_type_helper_oneapi.h"
#include "algorithms/threading/threading.h"
#include "algorithms/kernel/service_error_handling.h"
#include "algorithms/kernel/service_sort.h"
#include "algorithms/kernel/dtrees/service_array.h"
#include "service/kernel/service_arrays.h"
#include "externals/service_memory.h"
#include "service/kernel/service_data_utils.h"
#include "service/kernel/data_management/service_numeric_table.h"

#include "algorithms/kernel/dtrees/forest/oneapi/cl_kernels/df_common_kernels.cl"

#include "oneapi/internal/execution_context.h"
#include "oneapi/internal/types.h"

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
    typedef int IndexType; // TODO: should be unsigned int

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
    IndexedFeaturesOneAPI() : _data(), _sizeOfIndex(sizeof(IndexType)), _nCols(0), _nRows(0), _capacity(0), _maxNumIndices(0) {}
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
                                   oneapi::internal::UniversalBuffer & indices, int featureId, int nFeatures, int nRows);

    services::Status collectBinBorders(oneapi::internal::UniversalBuffer & values, oneapi::internal::UniversalBuffer & binOffsets,
                                       oneapi::internal::UniversalBuffer & binBorders, int nRows, int maxBins);

    services::Status computeBins(oneapi::internal::UniversalBuffer & values, oneapi::internal::UniversalBuffer & indices,
                                 oneapi::internal::UniversalBuffer & binBorders, oneapi::internal::UniversalBuffer & bins, int nRows, int nBins,
                                 int localSize, int nLocalBlocks);

    services::Status computeBins(oneapi::internal::UniversalBuffer & values, oneapi::internal::UniversalBuffer & indices,
                                 oneapi::internal::UniversalBuffer & bins, FeatureEntry & entry, int nRows,
                                 const dtrees::internal::BinParams * pBinPrm);

    services::Status makeIndex(const services::Buffer<algorithmFPType> & data, int featureId, int nFeatures, int nRows,
                               const dtrees::internal::BinParams * pBinPrm, oneapi::internal::UniversalBuffer & bins, FeatureEntry & entry);

    services::Status storeColumn(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & fullData, int featureId,
                                 int nFeatures, int nRows);

protected:
    services::Collection<oneapi::internal::UniversalBuffer> _data;
    oneapi::internal::UniversalBuffer _fullData;
    oneapi::internal::UniversalBuffer _binOffsets;
    daal::internal::TArray<FeatureEntry, sse2> _entries;
    size_t _sizeOfIndex;
    size_t _nRows;
    size_t _nCols;
    size_t _capacity;
    size_t _maxNumIndices;
    IndexType _totalBins;

    oneapi::internal::UniversalBuffer _values;
    oneapi::internal::UniversalBuffer _values_buf;
    oneapi::internal::UniversalBuffer _indices;
    oneapi::internal::UniversalBuffer _indices_buf;

    const uint32_t _maxWorkItemsPerGroup = 128;   // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer       = 30000; // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup   = 16;    // preferable maximal sub-group size
};

class TreeNodeStorage
{
public:
    TreeNodeStorage() {}

    oneapi::internal::UniversalBuffer & getHistograms() { return _histogramsForFeatures; }

    void clear() { _histogramsForFeatures = oneapi::internal::UniversalBuffer(); }

    template <typename algorithmFPType>
    services::Status allocate(const decision_forest::internal::IndexedFeaturesOneAPI<algorithmFPType> & indexedFeatures);

private:
    oneapi::internal::UniversalBuffer _histogramsForFeatures;
};

} /* namespace internal */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
