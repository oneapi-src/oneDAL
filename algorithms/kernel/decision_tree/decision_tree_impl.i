/* file: decision_tree_impl.i */
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
        return static_cast<T *>(daal_calloc(size * sizeof(T), alignment));
    }
};

template <>
class allocation<void>
{
public:
    static DAAL_FORCEINLINE void * malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT) { return daal_calloc(size, alignment); }
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
        : _size(table.getNumberOfColumns()), _types(daal_alloc<data_management::features::FeatureType>(_size))
    {
        for (FeatureIndex i = 0; i < _size; ++i)
        {
            _types[i] = table.getFeatureType(i);
        }
    }

    ~FeatureTypesCache()
    {
        daal_free(_types);
        _types = nullptr;
    }

    FeatureTypesCache(const FeatureTypesCache &) = delete;
    FeatureTypesCache & operator=(const FeatureTypesCache &) = delete;

    data_management::features::FeatureType operator[](FeatureIndex index) const
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
