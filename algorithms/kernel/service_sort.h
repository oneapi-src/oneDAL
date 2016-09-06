/* file: service_sort.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
            daal::swap<algorithmDataType, cpu>(x[k], x[l + 1]);
            if(x[l] > x[ir])
            {
                daal::swap<algorithmDataType, cpu>(x[l], x[ir]);
            }
            if(x[l + 1] > x[ir])
            {
                daal::swap<algorithmDataType, cpu>(x[l + 1], x[ir]);
            }
            if(x[l] > x[l + 1])
            {
                daal::swap<algorithmDataType, cpu>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for(;;)
            {
                while(x[++i] < a);
                while(x[--j] > a);
                if(j < i) { break; }
                daal::swap<algorithmDataType, cpu>(x[i], x[j]);
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
            daal::swap<algorithmDataType, cpu>(x[k], x[l + 1]);

            if(compare(x + l, x + ir) == 1)
            {
                daal::swap<algorithmDataType, cpu>(x[l], x[ir]);
            }
            if(compare(x + l + 1, x + ir) == 1)
            {
                daal::swap<algorithmDataType, cpu>(x[l + 1], x[ir]);
            }
            if(compare(x + l, x + l + 1) == 1)
            {
                daal::swap<algorithmDataType, cpu>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for(;;)
            {
                while(compare(&x[++i], &a) == -1);
                while(compare(&x[--j], &a) ==  1);
                if(j < i) { break; }
                daal::swap<algorithmDataType, cpu>(x[i], x[j]);
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
            daal::swap<algorithmDataType,  cpu>(x[k], x[l + 1]);
            daal::swap<algorithmIndexType, cpu>(index[k], index[l + 1]);
            if(x[l] > x[ir])
            {
                daal::swap<algorithmDataType,  cpu>(x[l], x[ir]);
                daal::swap<algorithmIndexType, cpu>(index[l], index[ir]);
            }
            if(x[l + 1] > x[ir])
            {
                daal::swap<algorithmDataType,  cpu>(x[l + 1], x[ir]);
                daal::swap<algorithmIndexType, cpu>(index[l + 1], index[ir]);
            }
            if(x[l] > x[l + 1])
            {
                daal::swap<algorithmDataType,  cpu>(x[l], x[l + 1]);
                daal::swap<algorithmIndexType, cpu>(index[l], index[l + 1]);
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
                daal::swap<algorithmDataType,  cpu>(x[i], x[j]);
                daal::swap<algorithmIndexType, cpu>(index[i], index[j]);
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
            daal::swap<algorithmFPtype, cpu>(x[k], x[l + 1]);
            daal::swap<wType, cpu>(w[k], w[l + 1]);
            daal::swap<zType, cpu>(z[k], z[l + 1]);
            if(x[l] > x[ir])
            {
                daal::swap<algorithmFPtype, cpu>(x[l], x[ir]);
                daal::swap<wType, cpu>(w[l], w[ir]);
                daal::swap<zType, cpu>(z[l], z[ir]);
            }
            if(x[l + 1] > x[ir])
            {
                daal::swap<algorithmFPtype, cpu>(x[l + 1], x[ir]);
                daal::swap<wType, cpu>(w[l + 1], w[ir]);
                daal::swap<zType, cpu>(z[l + 1], z[ir]);
            }
            if(x[l] > x[l + 1])
            {
                daal::swap<algorithmFPtype, cpu>(x[l], x[l + 1]);
                daal::swap<wType, cpu>(w[l], w[l + 1]);
                daal::swap<zType, cpu>(z[l], z[l + 1]);
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
                daal::swap<algorithmFPtype, cpu>(x[i], x[j]);
                daal::swap<wType, cpu>(w[i], w[j]);
                daal::swap<zType, cpu>(z[i], z[j]);
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

}
}
}

#endif
