/* file: dtrees_feature_type_helper.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of a service class that provides optimal access to the feature types
//--
*/

#ifndef __DTREES_FEATURE_TYPE_HELPER_H__
#define __DTREES_FEATURE_TYPE_HELPER_H__

#include "service_memory.h"
#include "service_numeric_table.h"

typedef double ModelFPType;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// Helper class, provides optimal access to the feature types
//////////////////////////////////////////////////////////////////////////////////////////
class FeatureTypes
{
public:
    FeatureTypes(): _bAllUnordered(false){}
    ~FeatureTypes();
    bool init(const NumericTable& data);

    bool isUnordered(size_t iFeature) const
    {
        return _bAllUnordered || (_aFeat && findInBuf(iFeature));
    }

    bool hasUnorderedFeatures() const { return (_bAllUnordered || _nNoOrderedFeat); }

    size_t getNumberOfFeatures() const
    {
        return _nFeat;
    }

private:
    void allocBuf(size_t n);
    void destroyBuf();
    bool findInBuf(size_t iFeature) const;

private:
    bool* _aFeat = nullptr; //buffer with minimal required features data
    size_t _nFeat = 0; //size of the buffer
    size_t _nNoOrderedFeat = 0;
    bool _bAllUnordered = false;
    int _firstUnordered = -1;
    int _lastUnordered = -1;
};

struct BinParams
{
    BinParams(size_t _maxBins, size_t _minBinSize) : maxBins(_maxBins), minBinSize(_minBinSize){}
    BinParams(const BinParams& o) : maxBins(o.maxBins), minBinSize(o.minBinSize){}

    size_t maxBins = 256;
    size_t minBinSize = 5;
};

//////////////////////////////////////////////////////////////////////////////////////////
// IndexedFeatures. Creates and stores index of every feature
// Sorts every feature and creates the mapping: features value -> index of the value
// in the sorted array of unique values of the feature in increasing order
//////////////////////////////////////////////////////////////////////////////////////////
class IndexedFeatures
{
public:
    typedef unsigned int IndexType;
    struct FeatureEntry
    {
        DAAL_NEW_DELETE();
        IndexType    numIndices = 0; //number of indices or bins
        ModelFPType* binBorders = nullptr; //right bin borders

        services::Status allocBorders();
        ~FeatureEntry();
    };

public:
    IndexedFeatures() : _data(nullptr), _entries(nullptr), _nCols(0), _nRows(0), _capacity(0), _maxNumIndices(0){}
    ~IndexedFeatures();

    template <typename algorithmFPType, CpuType cpu>
    services::Status init(const NumericTable& nt, const FeatureTypes* featureTypes = nullptr,
        const BinParams* pBimPrm = nullptr);

    //get max number of indices for that feature
    IndexType numIndices(size_t iCol) const
    {
        return _entries[iCol].numIndices;
    }

    //get max number of indices among all features
    IndexType maxNumIndices() const { return _maxNumIndices;  }

    //returns true if the feature is mapped to bins
    bool isBinned(size_t iCol) const { DAAL_ASSERT(iCol < _nCols); return !!_entries[iCol].binBorders; }

    //returns right border of the bin if the feature is a binned one
    ModelFPType binRightBorder(size_t iCol, size_t iBin) const
    {
        DAAL_ASSERT(isBinned(iCol));
        DAAL_ASSERT(iBin < numIndices(iCol));
        return _entries[iCol].binBorders[iBin];
    }

    //for low-level optimization
    const IndexType* data(size_t iFeature) const { return _data + _nRows*iFeature; }

    size_t nRows() const { return _nRows; }
    size_t nCols() const { return _nCols; }

protected:
    services::Status alloc(size_t nCols, size_t nRows);

protected:
    IndexType* _data;
    FeatureEntry* _entries;
    size_t _nRows;
    size_t _nCols;
    size_t _capacity;
    size_t _maxNumIndices;
};

} /* namespace internal */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif
