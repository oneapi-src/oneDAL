/* file: kdtree_knn_impl.i */
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
//  Common functions for K-Nearest Neighbors
//--
*/

#ifndef __KDTREE_KNN_IMPL_I__
#define __KDTREE_KNN_IMPL_I__

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif


#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE __forceinline
    #define DAAL_FORCENOINLINE __declspec(noinline)
#else
    #define DAAL_FORCEINLINE inline __attribute__((always_inline))
    #define DAAL_FORCENOINLINE __attribute__((noinline))
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #define DAAL_PREFETCH_READ_T0(addr) _mm_prefetch((char *)addr, _MM_HINT_T0)
#else
    #define DAAL_PREFETCH_READ_T0(addr) __builtin_prefetch((char *)addr, 0, 3)
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
#define __KDTREE_LEAF_BUCKET_SIZE 31 // Must be ((power of 2) minus 1).
#define __KDTREE_FIRST_PART_LEAF_NODES_PER_THREAD 3
#define __KDTREE_DIMENSION_SELECTION_SIZE 128
#define __KDTREE_MEDIAN_RANDOM_SAMPLE_COUNT 1024
#define __KDTREE_DEPTH_MULTIPLICATION_FACTOR 4
#define __KDTREE_SEARCH_SKIP 32
#define __KDTREE_INDEX_VALUE_PAIRS_PER_THREAD 8192
#define __KDTREE_SAMPLES_PERCENT 0.5
#define __KDTREE_MAX_SAMPLES 1024
#define __KDTREE_MIN_SAMPLES 256
#define __SIMDWIDTH 8

#define __KDTREE_NULLDIMENSION (static_cast<size_t>(-1))

template <CpuType cpu, typename T>
inline const T & min(const T & a, const T & b) { return !(b < a) ? a : b; }

template <CpuType cpu, typename T>
inline const T & max(const T & a, const T & b) { return (a < b) ? b : a; }

template <typename algorithmFpType, CpuType cpu>
int compareFp(const void * p1, const void * p2)
{
    const algorithmFpType & v1 = *static_cast<const algorithmFpType *>(p1);
    const algorithmFpType & v2 = *static_cast<const algorithmFpType *>(p2);
    return (v1 < v2) ? -1 : 1;
}

template <typename T, CpuType cpu>
class Stack
{
public:
    Stack() : _data(nullptr) {}

    ~Stack() { services::daal_free(_data); }

    bool init(size_t size)
    {
        _data = static_cast<T *>(services::daal_malloc(size * sizeof(T)));
        _size = size;
        _top = _sizeMinus1 = size - 1;
        _count = 0;
        return _data;
    }

    void clear()
    {
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset()
    {
        _top = _sizeMinus1;
        _count = 0;
    }

    DAAL_FORCEINLINE void push(const T & value)
    {
        if (_count >= _size)
        {
            grow();
        }

        _top = (_top + 1) & _sizeMinus1;
        _data[_top] = value;
        ++_count;
    }

    DAAL_FORCEINLINE T pop()
    {
        const T value = _data[_top--];
        _top = _top & _sizeMinus1;
        --_count;
        return value;
    }

    bool empty() const { return (_count == 0); }

    size_t size() const { return _count; }

    void grow()
    {
        _size *= 2;
        T * const newData = static_cast<T *>(services::daal_malloc(_size * sizeof(T)));
        if (_top == _sizeMinus1)
        {
            _top = _size - 1;
        }
        _sizeMinus1 = _size - 1;
        services::daal_memcpy_s(newData, _size * sizeof(T), _data, _count * sizeof(T));
        T * const oldData = _data;
        _data = newData;
        services::daal_free(oldData);
    }

private:
    T * _data;
    size_t _top;
    size_t _count;
    size_t _size;
    size_t _sizeMinus1;
};

} // namespace internal
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
