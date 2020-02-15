/* file: kdtree_knn_impl.i */
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
//  Common functions for K-Nearest Neighbors
//--
*/

#ifndef __KDTREE_KNN_IMPL_I__
#define __KDTREE_KNN_IMPL_I__

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif

#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE   __forceinline
    #define DAAL_FORCENOINLINE __declspec(noinline)
#else
    #define DAAL_FORCEINLINE   inline __attribute__((always_inline))
    #define DAAL_FORCENOINLINE __attribute__((noinline))
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1900)
    #define DAAL_ALIGNAS(n) __declspec(align(n))
#else
    #define DAAL_ALIGNAS(n) alignas(n)
#endif

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace internal
{
#define __KDTREE_MAX_NODE_COUNT_MULTIPLICATION_FACTOR 3
#define __KDTREE_LEAF_BUCKET_SIZE                     31 // Must be ((power of 2) minus 1).
#define __KDTREE_FIRST_PART_LEAF_NODES_PER_THREAD     3
#define __KDTREE_DIMENSION_SELECTION_SIZE             128
#define __KDTREE_MEDIAN_RANDOM_SAMPLE_COUNT           1024
#define __KDTREE_DEPTH_MULTIPLICATION_FACTOR          4
#define __KDTREE_SEARCH_SKIP                          32
#define __KDTREE_INDEX_VALUE_PAIRS_PER_THREAD         8192
#define __KDTREE_SAMPLES_PERCENT                      0.5
#define __KDTREE_MAX_SAMPLES                          1024
#define __KDTREE_MIN_SAMPLES                          256
#define __SIMDWIDTH                                   8

#define __KDTREE_NULLDIMENSION (static_cast<size_t>(-1))

template <CpuType cpu, typename T>
inline const T & min(const T & a, const T & b)
{
    return !(b < a) ? a : b;
}

template <CpuType cpu, typename T>
inline const T & max(const T & a, const T & b)
{
    return (a < b) ? b : a;
}

template <typename T, CpuType cpu>
class Stack
{
private:
    T * _data;
    long _top;
    long _sizeMinus1;
    size_t _size;
    /*
        There was a strong assumption that stack can be initialized by only powers of 2.
        Due to the usage scenario it's true but wrong in general. This function should fix it.
    */
    size_t nextPowerOf2(size_t x)
    {
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x |= x >> 32;
        x++;
        return x;
    }

public:
    Stack() : _data(nullptr) {}

    void clear()
    {
        if (_data != nullptr)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    ~Stack() { clear(); }

    bool init(size_t size)
    {
        _size       = size ? nextPowerOf2(size) : 1; //Handles both cases size == 0 and size != 0
        _data       = static_cast<T *>(services::internal::service_malloc<T, cpu>(_size * sizeof(T)));
        _sizeMinus1 = (long)_size - 1;
        _top        = -1;
        return _data;
    }

    void reset() { _top = -1; }

    DAAL_FORCEINLINE services::Status push(const T & value)
    {
        services::Status status;
        if (_top >= _sizeMinus1)
        {
            status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }
        _data[++_top] = value;
        return status;
    }

    DAAL_FORCEINLINE T pop() { return _data[_top--]; }

    bool empty() const { return _top == -1; }

    size_t size() const { return (size_t)(_top + 1); }

    services::Status grow()
    {
        _size <<= 1; // Equal to multiplying by 2
        T * const newData = static_cast<T *>(services::internal::service_malloc<T, cpu>(_size * sizeof(T)));
        DAAL_CHECK_MALLOC(newData)
        _sizeMinus1 = (long)_size - 1;
        int result  = services::internal::daal_memcpy_s(newData, _size * sizeof(T), _data, size() * sizeof(T));
        T * oldData = _data;
        _data       = newData;
        services::daal_free(oldData);
        oldData = nullptr;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }
};

} // namespace internal
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
