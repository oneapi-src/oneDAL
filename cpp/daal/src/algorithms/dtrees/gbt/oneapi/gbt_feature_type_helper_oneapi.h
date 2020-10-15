/* file: gbt_feature_type_helper_oneapi.h */
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

/*
//++
//  Implementation of a service class that provides optimal access to the feature types
//--
*/

#ifndef __GBT_FEATURE_TYPE_HELPER_ONEAPI_H__
#define __GBT_FEATURE_TYPE_HELPER_ONEAPI_H__

#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/gbt/oneapi/gbt_feature_type_helper_oneapi.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/dtrees/service_array.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/data_management/service_numeric_table.h"

#include "src/algorithms/dtrees/gbt/oneapi/cl_kernels/gbt_common_kernels.cl"

#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/types.h"

namespace daal
{
namespace algorithms
{
namespace gbt
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
        services::internal::sycl::UniversalBuffer binBorders; //right bin borders

        services::Status allocBorders();
        ~FeatureEntry();
    };

public:
    IndexedFeaturesOneAPI() : _data(), _entries(nullptr), _sizeOfIndex(sizeof(IndexType)), _nCols(0), _nRows(0), _capacity(0), _maxNumIndices(0) {}
    ~IndexedFeaturesOneAPI();

    services::Status init(NumericTable & nt, const dtrees::internal::FeatureTypes * featureTypes, const dtrees::internal::BinParams * pBinPrm);

    //get max number of indices for that feature
    IndexType numIndices(size_t iCol) const { return _entries[iCol].numIndices; }

    IndexType totalBins() const { return _totalBins; }

    services::internal::sycl::UniversalBuffer & binBorders(size_t iCol) const { return _entries[iCol].binBorders; }

    services::internal::sycl::UniversalBuffer & binOffsets() { return _binOffsets; }

    services::internal::sycl::UniversalBuffer & getFullData() { return _fullData; }

    //for low-level optimization
    const services::internal::sycl::UniversalBuffer & getFeature(size_t iFeature) const { return _data[iFeature]; }

    size_t nRows() const { return _nRows; }
    size_t nCols() const { return _nCols; }

protected:
    services::Status alloc(size_t nCols, size_t nRows);

    services::Status extractColumn(const services::internal::Buffer<algorithmFPType> & data, services::internal::sycl::UniversalBuffer & values,
                                   services::internal::sycl::UniversalBuffer & indices, int featureId, int nFeatures, int nRows);

    services::Status radixScan(services::internal::sycl::UniversalBuffer & values, services::internal::sycl::UniversalBuffer & partialHists,
                               int nRows, int bitOffset, int localSize, int nLocalSums);

    services::Status radixHistScan(services::internal::sycl::UniversalBuffer & partialHists,
                                   services::internal::sycl::UniversalBuffer & partialPrefixHists, int nSubgroupSums, int localSize);

    services::Status radixReorder(services::internal::sycl::UniversalBuffer & valuesSrc, services::internal::sycl::UniversalBuffer & indicesSrc,
                                  services::internal::sycl::UniversalBuffer & partialPrefixHist,
                                  services::internal::sycl::UniversalBuffer & valuesDst, services::internal::sycl::UniversalBuffer & indicesDst,
                                  int nRows, int bitOffset, int localSize, int nLocalHists);

    services::Status radixSort(services::internal::sycl::UniversalBuffer & values, services::internal::sycl::UniversalBuffer & indices,
                               services::internal::sycl::UniversalBuffer & values_buf, services::internal::sycl::UniversalBuffer & indices_buf,
                               int nRows);

    services::Status collectBinBorders(services::internal::sycl::UniversalBuffer & values, services::internal::sycl::UniversalBuffer & binOffsets,
                                       services::internal::sycl::UniversalBuffer & binBorders, int nRows, int maxBins);

    services::Status computeBins(services::internal::sycl::UniversalBuffer & values, services::internal::sycl::UniversalBuffer & indices,
                                 services::internal::sycl::UniversalBuffer & binBorders, services::internal::sycl::UniversalBuffer & bins, int nRows,
                                 int nBins, int localSize, int nLocalBlocks);

    services::Status computeBins(services::internal::sycl::UniversalBuffer & values, services::internal::sycl::UniversalBuffer & indices,
                                 services::internal::sycl::UniversalBuffer & bins, FeatureEntry & entry, int nRows,
                                 const dtrees::internal::BinParams * pBinPrm);

    services::Status makeIndex(const services::internal::Buffer<algorithmFPType> & data, int featureId, int nFeatures, int nRows,
                               const dtrees::internal::BinParams * pBinPrm, services::internal::sycl::UniversalBuffer & bins, FeatureEntry & entry);

    services::Status storeColumn(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & fullData,
                                 int featureId, int nFeatures, int nRows);

protected:
    services::Collection<services::internal::sycl::UniversalBuffer> _data;
    services::internal::sycl::UniversalBuffer _fullData;
    services::internal::sycl::UniversalBuffer _binOffsets;
    FeatureEntry * _entries;
    size_t _sizeOfIndex;
    size_t _nRows;
    size_t _nCols;
    size_t _capacity;
    size_t _maxNumIndices;
    IndexType _totalBins;

    services::internal::sycl::UniversalBuffer _values;
    services::internal::sycl::UniversalBuffer _values_buf;
    services::internal::sycl::UniversalBuffer _indices;
    services::internal::sycl::UniversalBuffer _indices_buf;

    const uint32_t _maxWorkItemsPerGroup = 128;   // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer       = 30000; // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup   = 16;    // preferable maximal sub-group size
    const uint32_t _radixBits            = 4;
};

class TreeNodeStorage
{
public:
    TreeNodeStorage() {}

    services::internal::sycl::UniversalBuffer & getHistograms() { return _histogramsForFeatures; }

    void clear() { _histogramsForFeatures = services::internal::sycl::UniversalBuffer(); }

    template <typename algorithmFPType>
    services::Status allocate(const gbt::internal::IndexedFeaturesOneAPI<algorithmFPType> & indexedFeatures);

private:
    services::internal::sycl::UniversalBuffer _histogramsForFeatures;
};

template <typename algorithmFPType>
struct BestSplitOneAPI
{
    BestSplitOneAPI();

    algorithmFPType _impurityDecrease;
    int _featureIndex;
    int _featureValue;
    algorithmFPType _leftGTotal;
    algorithmFPType _leftHTotal;
    algorithmFPType _rightGTotal;
    algorithmFPType _rightHTotal;
};

} /* namespace internal */
} // namespace gbt
} /* namespace algorithms */
} /* namespace daal */

#endif
