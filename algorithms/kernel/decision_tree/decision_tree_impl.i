/* file: decision_tree_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Common functions for Decision tree
//--
*/

#ifndef __DECISION_TREE_IMPL_I__
#define __DECISION_TREE_IMPL_I__

#include "services/daal_memory.h"
#include "services/env_detect.h"
#include "numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace internal
{

using namespace daal::services;

template <typename T>
class allocation
{
public:
    static DAAL_FORCEINLINE T * malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
    {
        return static_cast<T *>(daal_malloc(size * sizeof(T), alignment));
    }
};

template <>
class allocation<void>
{
public:
    static DAAL_FORCEINLINE void * malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
    {
        return daal_malloc(size, alignment);
    }
};

template <typename T>
DAAL_FORCEINLINE T * daal_alloc(size_t size, size_t alignment)
{
    return allocation<T>::malloc(size, alignment);
}

template <typename T>
DAAL_FORCEINLINE T * daal_alloc(size_t size)
{
    return allocation<T>::malloc(size);
}

typedef size_t FeatureIndex;

class FeatureTypesCache
{
public:
    FeatureTypesCache(const data_management::NumericTable & table)
        : _size(table.getNumberOfColumns()),
          _types(daal_alloc<data_management::features::FeatureType>(_size))
    {
        for (FeatureIndex i = 0; i < _size; ++i)
        {
            _types[i] = table.getFeatureType(i);
        }
    }

    ~FeatureTypesCache()
    {
        daal_free(_types);
    }

    FeatureTypesCache(const FeatureTypesCache &) = delete;
    FeatureTypesCache & operator= (const FeatureTypesCache &) = delete;

    data_management::features::FeatureType operator[] (FeatureIndex index) const
    {
        DAAL_ASSERT(index < _size);
        return _types[index];
    }

private:
    size_t _size;
    data_management::features::FeatureType * _types;
};

} // namespace internal
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
