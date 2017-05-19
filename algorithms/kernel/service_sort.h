/* file: service_sort.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of sorting algorithms.
//--
*/

#ifndef __SERVICE_SORT_H__
#define __SERVICE_SORT_H__

#include "service_utils.h"
#include "services/collection.h"

#if defined(__INTEL_COMPILER_BUILD_DATE)
#include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace internal
{

typedef int (*CompareFunction)(const void *, const void *);

/**
 * \brief Quick sort function that sorts array x
 *
 * \param n[in]     Length of input arrays
 * \param x[in,out] Array to sort
 */
template <typename algorithmDataType, CpuType cpu>
void qSort(size_t n, algorithmDataType *x)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = n - 1;

    for(;;)
    {
        if(ir - l < M)
        {
            for(j = l + 1; j <= ir; j++)
            {
                a = x[j];

                for(i = j - 1; i >= l; i--)
                {
                    if(x[i] <= a) { break; }
                    x[i + 1] = x[i];
                }

                x[i + 1] = a;
            }

            if(jstack < 0) { break; }

            ir = istack[jstack--];
            l = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmDataType>(x[k], x[l + 1]);
            if(x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[ir]);
            }
            if(x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l + 1], x[ir]);
            }
            if(x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for(;;)
            {
                while(x[++i] < a);
                while(x[--j] > a);
                if(j < i) { break; }
                daal::services::internal::swap<cpu, algorithmDataType>(x[i], x[j]);
            }
            x[l + 1] = x[j];

            x[j] = a;
            jstack += 2;

            if(ir - i + 1 >= j - l)
            {
                istack[jstack  ] = ir ;
                istack[jstack - 1] = i  ;
                ir = j - 1;
            }
            else
            {
                istack[jstack  ] = j - 1;
                istack[jstack - 1] = l  ;
                l = i;
            }
        }
    }
}

/**
 * \brief Quick sort function that sorts array x
 *
 * \param n[in]       Length of input arrays
 * \param x[in,out]   Array to sort
 * \param compare[in] Pointer to compare function
 */
template <typename algorithmDataType, CpuType cpu>
void qSort(size_t n, algorithmDataType *x, CompareFunction compare)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = n - 1;

    for(;;)
    {
        if(ir - l < M)
        {
            for(j = l + 1; j <= ir; j++)
            {
                a = x[j];

                for(i = j - 1; i >= l; i--)
                {
                    if(compare(x + i, &a) < 1) { break; }
                    x[i + 1] = x[i];
                }

                x[i + 1] = a;
            }

            if(jstack < 0) { break; }

            ir = istack[jstack--];
            l = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmDataType>(x[k], x[l + 1]);

            if(compare(x + l, x + ir) == 1)
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[ir]);
            }
            if(compare(x + l + 1, x + ir) == 1)
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l + 1], x[ir]);
            }
            if(compare(x + l, x + l + 1) == 1)
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for(;;)
            {
                while(compare(&x[++i], &a) == -1);
                while(compare(&x[--j], &a) ==  1);
                if(j < i) { break; }
                daal::services::internal::swap<cpu, algorithmDataType>(x[i], x[j]);
            }
            x[l + 1] = x[j];

            x[j] = a;
            jstack += 2;

            if(ir - i + 1 >= j - l)
            {
                istack[jstack  ] = ir ;
                istack[jstack - 1] = i  ;
                ir = j - 1;
            }
            else
            {
                istack[jstack  ] = j - 1;
                istack[jstack - 1] = l  ;
                l = i;
            }
        }
    }
}


/**
 *  \brief Quick sort function that sorts array x and rearranges array index
 *         accordingly
 *
 *  \param n[in] Length of input arrays
 *  \param x     Array that is used as "key" when sorted
 *  \param index Array that is used as "value" when sorted
 */
template <typename algorithmDataType, typename algorithmIndexType, CpuType cpu>
void qSort(size_t n, algorithmDataType *x, algorithmIndexType *index)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    algorithmIndexType b;
    const int M = 7, NSTACK = 128;
    algorithmDataType istack[NSTACK];

    ir = n - 1;

    for(;;)
    {
        if(ir - l < M)
        {
            for(j = l + 1; j <= ir; j++)
            {
                a = x[j];
                b = index[j];

                for(i = j - 1; i >= l; i--)
                {
                    if(x[i] <= a) { break; }
                    x[i + 1] = x[i];
                    index[i + 1] = index[i];
                }

                x[i + 1] = a;
                index[i + 1] = b;
            }

            if(jstack < 0) { break; }

            ir = istack[jstack--];
            l = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmDataType>(x[k], x[l + 1]);
            daal::services::internal::swap<cpu, algorithmIndexType>(index[k], index[l + 1]);
            if(x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[ir]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[l], index[ir]);
            }
            if(x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l + 1], x[ir]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[l + 1], index[ir]);
            }
            if(x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[l + 1]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[l], index[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            b = index[l + 1];
            for(;;)
            {
                while(x[++i] < a);
                while(x[--j] > a);
                if(j < i) { break; }
                daal::services::internal::swap<cpu, algorithmDataType>(x[i], x[j]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[i], index[j]);
            }
            x[l + 1] = x[j];
            index[l + 1] = index[j];

            x[j] = a;
            index[j] = b;
            jstack += 2;

            if(ir - i + 1 >= j - l)
            {
                istack[jstack  ] = ir ;
                istack[jstack - 1] = i  ;
                ir = j - 1;
            }
            else
            {
                istack[jstack  ] = j - 1;
                istack[jstack - 1] = l  ;
                l = i;
            }
        }
    }
}

template <typename algorithmFPtype, typename wType, typename zType, CpuType cpu>
void qSort(size_t n, algorithmFPtype *x, wType *w, zType *z)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmFPtype a;
    wType b;
    zType c;
    const int M = 7, NSTACK = 128;
    algorithmFPtype istack[NSTACK];

    ir = n - 1;

    for(;;)
    {
        if(ir - l < M)
        {
            for(j = l + 1; j <= ir; j++)
            {
                a = x[j];
                b = w[j];
                c = z[j];

                for(i = j - 1; i >= l; i--)
                {
                    if(x[i] <= a) { break; }
                    x[i + 1] = x[i];
                    w[i + 1] = w[i];
                    z[i + 1] = z[i];
                }

                x[i + 1] = a;
                w[i + 1] = b;
                z[i + 1] = c;
            }

            if(jstack < 0) { break; }

            ir = istack[jstack--];
            l = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmFPtype>(x[k], x[l + 1]);
            daal::services::internal::swap<cpu, wType>(w[k], w[l + 1]);
            daal::services::internal::swap<cpu, zType>(z[k], z[l + 1]);
            if(x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l], x[ir]);
                daal::services::internal::swap<cpu, wType>(w[l], w[ir]);
                daal::services::internal::swap<cpu, zType>(z[l], z[ir]);
            }
            if(x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l + 1], x[ir]);
                daal::services::internal::swap<cpu, wType>(w[l + 1], w[ir]);
                daal::services::internal::swap<cpu, zType>(z[l + 1], z[ir]);
            }
            if(x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l], x[l + 1]);
                daal::services::internal::swap<cpu, wType>(w[l], w[l + 1]);
                daal::services::internal::swap<cpu, zType>(z[l], z[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            b = w[l + 1];
            c = z[l + 1];
            for(;;)
            {
                while(x[++i] < a);
                while(x[--j] > a);
                if(j < i) { break; }
                daal::services::internal::swap<cpu, algorithmFPtype>(x[i], x[j]);
                daal::services::internal::swap<cpu, wType>(w[i], w[j]);
                daal::services::internal::swap<cpu, zType>(z[i], z[j]);
            }
            x[l + 1] = x[j];
            w[l + 1] = w[j];
            z[l + 1] = z[j];

            x[j] = a;
            w[j] = b;
            z[j] = c;
            jstack += 2;

            if(ir - i + 1 >= j - l)
            {
                istack[jstack  ] = ir ;
                istack[jstack - 1] = i  ;
                ir = j - 1;
            }
            else
            {
                istack[jstack  ] = j - 1;
                istack[jstack - 1] = l  ;
                l = i;
            }
        }
    }
}

template <typename algorithmFPtype, CpuType cpu>
void indexBubbleSortDesc(services::Collection<algorithmFPtype> &x, services::Collection<algorithmFPtype> &idx1, services::Collection<algorithmFPtype> &idx2)
{
    size_t n = x.size();
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n - i - 1; j++)
        {
            if (x[j] < x[j + 1])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[j], x[j + 1]);
                daal::services::internal::swap<cpu, algorithmFPtype>(idx1[j], idx1[j + 1]);
                daal::services::internal::swap<cpu, algorithmFPtype>(idx2[j], idx2[j + 1]);
            }
        }
    }
}

template <CpuType cpu, typename T>
struct RadixSort {};

#if defined(__INTEL_COMPILER_BUILD_DATE)
#define __RADIX_SORT_CAST32(x) (_castf32_u32(x))
#define __RADIX_SORT_CAST64(x) (_castf64_u64(x))
#else
#define __RADIX_SORT_CAST32(x) (*reinterpret_cast<const unsigned int *>(&(x)))
#define __RADIX_SORT_CAST64(x) (*reinterpret_cast<const DAAL_UINT64 *>(&(x)))
#endif

template <CpuType cpu>
struct RadixSort<cpu, float>
{
    template <typename T, typename Get>
    static void sort(T * in, size_t count, T * out, Get get)
    {
        using namespace daal::services::internal;

        typedef unsigned int IntegerType;
        typedef size_t CounterType;

        CounterType h[256], hps[257];
        const size_t hSize = sizeof(h) / sizeof(h[0]);
        T * first = in;
        T * second = out;
        const size_t count4 = count / 4 * 4;

        for (unsigned int pass = 0; pass < 3; ++pass)
        {
            for (size_t i = 0; i < hSize; ++i) { h[i] = 0; }

            for (size_t i = 0; i < count4; i += 4)
            {
                IntegerType val1 = __RADIX_SORT_CAST32(get(first[i]));
                IntegerType val2 = __RADIX_SORT_CAST32(get(first[i + 1]));
                IntegerType val3 = __RADIX_SORT_CAST32(get(first[i + 2]));
                IntegerType val4 = __RADIX_SORT_CAST32(get(first[i + 3]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
                ++h[(val2 >> (pass * 8)) & 0xFF];
                ++h[(val3 >> (pass * 8)) & 0xFF];
                ++h[(val4 >> (pass * 8)) & 0xFF];
            }
            for (size_t i = count4; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST32(get(first[i]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
            }

            CounterType sum = 0, prevSum = 0;
            for (size_t i = 0; i < hSize; ++i)
            {
                sum += h[i];
                hps[i] = prevSum;
                prevSum = sum;
            }
            hps[hSize] = prevSum;

            for (size_t i = 0; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST32(get(first[i]));
                const CounterType pos = hps[(val1 >> (pass * 8)) & 0xFF]++;
                second[pos] = first[i];
            }

            swap<cpu>(first, second);
        }
        {
            unsigned int pass = 3;
            for (size_t i = 0; i < hSize; ++i) { h[i] = 0; }

            for (size_t i = 0; i < count4; i += 4)
            {
                IntegerType val1 = __RADIX_SORT_CAST32(get(first[i]));
                IntegerType val2 = __RADIX_SORT_CAST32(get(first[i + 1]));
                IntegerType val3 = __RADIX_SORT_CAST32(get(first[i + 2]));
                IntegerType val4 = __RADIX_SORT_CAST32(get(first[i + 3]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
                ++h[(val2 >> (pass * 8)) & 0xFF];
                ++h[(val3 >> (pass * 8)) & 0xFF];
                ++h[(val4 >> (pass * 8)) & 0xFF];
            }
            for (size_t i = count4; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST32(get(first[i]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
            }

            CounterType sum = 0, prevSum = 0;
            for (size_t i = 0; i < hSize; ++i)
            {
                sum += h[i];
                hps[i] = prevSum;
                prevSum = sum;
            }
            hps[hSize] = prevSum;

            // Handle negative values.
            const size_t indexOfNegatives = hSize / 2;
            CounterType countOfNegatives = hps[hSize] - hps[indexOfNegatives];
            // Fixes offsets for positive values.
            for (size_t i = 0; i < indexOfNegatives - 1; ++i)
            {
                hps[i] += countOfNegatives;
            }
            // Fixes offsets for negative values.
            hps[hSize - 1] = h[hSize - 1];
            for (size_t i = 0; i < indexOfNegatives - 1; ++i)
            {
                hps[hSize - 2 - i] = hps[hSize - 1 - i] + h[hSize - 2 - i];
            }

            for (size_t i = 0; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST32(get(first[i]));
                const size_t bin = (val1 >> (pass * 8)) & 0xFF;
                size_t pos;
                if (bin >= indexOfNegatives) { pos = --hps[bin]; }
                else { pos = hps[bin]++; }
                second[pos] = first[i];
            }
        }
    }
};

template <CpuType cpu>
struct RadixSort<cpu, double>
{
    template <typename T, typename Get>
    static void sort(T * in, size_t count, T * out, Get get)
    {
        using namespace daal::services::internal;

        typedef DAAL_UINT64 IntegerType;
        typedef size_t CounterType;

        CounterType h[256], hps[257];
        const size_t hSize = sizeof(h) / sizeof(h[0]);
        T * first = in;
        T * second = out;
        const size_t count4 = count / 4 * 4;

        for (unsigned int pass = 0; pass < 7; ++pass)
        {
            for (size_t i = 0; i < hSize; ++i) { h[i] = 0; }

            for (size_t i = 0; i < count4; i += 4)
            {
                IntegerType val1 = __RADIX_SORT_CAST64(get(first[i]));
                IntegerType val2 = __RADIX_SORT_CAST64(get(first[i + 1]));
                IntegerType val3 = __RADIX_SORT_CAST64(get(first[i + 2]));
                IntegerType val4 = __RADIX_SORT_CAST64(get(first[i + 3]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
                ++h[(val2 >> (pass * 8)) & 0xFF];
                ++h[(val3 >> (pass * 8)) & 0xFF];
                ++h[(val4 >> (pass * 8)) & 0xFF];
            }
            for (size_t i = count4; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST64(get(first[i]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
            }

            CounterType sum = 0, prevSum = 0;
            for (size_t i = 0; i < hSize; ++i)
            {
                sum += h[i];
                hps[i] = prevSum;
                prevSum = sum;
            }
            hps[hSize] = prevSum;

            for (size_t i = 0; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST64(get(first[i]));
                const CounterType pos = hps[(val1 >> (pass * 8)) & 0xFF]++;
                second[pos] = first[i];
            }

            swap<cpu>(first, second);
        }
        {
            unsigned int pass = 7;
            for (size_t i = 0; i < hSize; ++i) { h[i] = 0; }

            for (size_t i = 0; i < count4; i += 4)
            {
                IntegerType val1 = __RADIX_SORT_CAST64(get(first[i]));
                IntegerType val2 = __RADIX_SORT_CAST64(get(first[i + 1]));
                IntegerType val3 = __RADIX_SORT_CAST64(get(first[i + 2]));
                IntegerType val4 = __RADIX_SORT_CAST64(get(first[i + 3]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
                ++h[(val2 >> (pass * 8)) & 0xFF];
                ++h[(val3 >> (pass * 8)) & 0xFF];
                ++h[(val4 >> (pass * 8)) & 0xFF];
            }
            for (size_t i = count4; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST64(get(first[i]));
                ++h[(val1 >> (pass * 8)) & 0xFF];
            }

            CounterType sum = 0, prevSum = 0;
            for (size_t i = 0; i < hSize; ++i)
            {
                sum += h[i];
                hps[i] = prevSum;
                prevSum = sum;
            }
            hps[hSize] = prevSum;

            // Handle negative values.
            const size_t indexOfNegatives = hSize / 2;
            CounterType countOfNegatives = hps[hSize] - hps[indexOfNegatives];
            // Fixes offsets for positive values.
            for (size_t i = 0; i < indexOfNegatives - 1; ++i)
            {
                hps[i] += countOfNegatives;
            }
            // Fixes offsets for negative values.
            hps[hSize - 1] = h[hSize - 1];
            for (size_t i = 0; i < indexOfNegatives - 1; ++i)
            {
                hps[hSize - 2 - i] = hps[hSize - 1 - i] + h[hSize - 2 - i];
            }

            for (size_t i = 0; i < count; ++i)
            {
                IntegerType val1 = __RADIX_SORT_CAST64(get(first[i]));
                const size_t bin = (val1 >> (pass * 8)) & 0xFF;
                size_t pos;
                if (bin >= indexOfNegatives) { pos = --hps[bin]; }
                else { pos = hps[bin]++; }
                second[pos] = first[i];
            }
        }
    }
};

template <CpuType cpu, typename U, typename T, typename Get>
inline void radixSort(T * in, size_t count, T * out, Get get)
{
    RadixSort<cpu, U>::sort(in, count, out, get);
}

}
}
}

#endif
