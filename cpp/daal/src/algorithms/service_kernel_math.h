/* file: service_kernel_math.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of math functions.
//--
*/

#ifndef __SERVICE_KERNEL_MATH_H__
#define __SERVICE_KERNEL_MATH_H__

#include "services/daal_defines.h"

#include "services/env_detect.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "data_management/features/defines.h"
#include "src/algorithms/service_error_handling.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/features/defines.h"
#include "immintrin.h"

#include "services/error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_arrays.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename FPType, CpuType cpu>
FPType distancePow2(const FPType * a, const FPType * b, size_t dim)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += (b[i] - a[i]) * (b[i] - a[i]);
    }

    return sum;
}

template <typename FPType, CpuType cpu>
FPType distancePow(const FPType * a, const FPType * b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return sum;
}

template <typename FPType, CpuType cpu>
FPType distance(const FPType * a, const FPType * b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return daal::internal::Math<FPType, cpu>::sPowx(sum, (FPType)1.0 / p);
}

template <typename FPType, CpuType cpu>
class PairwiseDistances
{
public:
    virtual ~PairwiseDistances() {};

    virtual services::Status init() = 0;

    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res) = 0;
    // virtual services::Status computeBatch(size_t aOffset, size_t aSize, size_t bOffset, size_t bSize, FPType * const res) = 0;
    // virtual services::Status computeFull(FPType * const res)                                                              = 0;
};

// compute: sum(A^2, 2) + sum(B^2, 2) -2*A*B'
template <typename FPType, CpuType cpu>
class EuclideanDistances : public PairwiseDistances<FPType, cpu>
{
public:
    EuclideanDistances(const NumericTable & a, const NumericTable & b, bool squared = true) : _a(a), _b(b), _squared(squared) {}

    virtual ~EuclideanDistances() {}

    virtual services::Status init()
    {
        services::Status s;

        normBufferA.reset(_a.getNumberOfRows());
        DAAL_CHECK_MALLOC(normBufferA.get());
        s |= computeNorm(_a, normBufferA.get());

        if (&_a != &_b)
        {
            normBufferB.reset(_b.getNumberOfRows());
            DAAL_CHECK_MALLOC(normBufferB.get());
            s |= computeNorm(_b, normBufferB.get());
        }

        return s;
    }

    // output:  Row-major matrix of size { aSize x bSize }
    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)
    {
        const size_t nRowsA = aSize;
        const size_t nColsA = _a.getNumberOfColumns();
        const size_t nRowsB = bSize;

        const size_t nRowsC = nRowsA;
        const size_t nColsC = nRowsB;

        computeABt(a, b, nRowsA, nColsA, nRowsB, res);

        const FPType * const aa = normBufferA.get() + aOffset;
        const FPType * const bb = (&_a == &_b) ? normBufferA.get() + bOffset : normBufferB.get() + bOffset;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nRowsC; i++)
        {
            for (size_t j = 0; j < nColsC; j++)
            {
                res[i * nColsC + j] = aa[i] + bb[j] - 2 * res[i * nColsC + j];
            }
        }

        if (!_squared)
        {
            daal::internal::Math<FPType, cpu> math;
            daal::services::internal::TArray<FPType, cpu> tmpArr(nRowsC * nColsC);
            FPType * tmp = tmpArr.get();
            math.vSqrt(nRowsC * nColsC, res, tmp);

            services::internal::daal_memcpy_s(res, nRowsC * nColsC * sizeof(FPType), tmp, nRowsC * nColsC * sizeof(FPType));
        }

        return services::Status();
    }

    // output:  Row-major matrix of size { aSize x bSize }
    virtual services::Status computeBatch(size_t aOffset, size_t aSize, size_t bOffset, size_t bSize, FPType * const res)
    {
        ReadRows<FPType, cpu> aDataRows(const_cast<NumericTable *>(&_a), aOffset, aSize);
        DAAL_CHECK_BLOCK_STATUS(aDataRows);
        const FPType * const aData = aDataRows.get();

        ReadRows<FPType, cpu> bDataRows(const_cast<NumericTable *>(&_b), bOffset, bSize);
        DAAL_CHECK_BLOCK_STATUS(bDataRows);
        const FPType * const bData = bDataRows.get();

        return computeBatch(aData, bData, aOffset, aSize, bOffset, bSize, res);
    }

    // output:  Row-major matrix of size { nrows(A) x nrows(B) }
    virtual services::Status computeFull(FPType * const res)
    {
        SafeStatus safeStat;

        const size_t nRowsA    = _a.getNumberOfRows();
        const size_t blockSize = 256;
        const size_t nBlocks   = nRowsA / blockSize + (nRowsA % blockSize > 0);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t i1    = iBlock * blockSize;
            const size_t i2    = (iBlock + 1 == nBlocks ? nRowsA : i1 + blockSize);
            const size_t iSize = i2 - i1;

            const size_t nRowsB = _b.getNumberOfRows();

            DAAL_CHECK_STATUS_THR(computeBatch(i1, iSize, 0, nRowsB, res + i1 * nRowsB));
        });

        return safeStat.detach();
    }

protected:
    // compute (sum(A^2, 2))
    services::Status computeNorm(const NumericTable & ntData, FPType * const res)
    {
        const size_t nRows = ntData.getNumberOfRows();
        const size_t nCols = ntData.getNumberOfColumns();

        const size_t blockSize = 512;
        const size_t nBlocks   = nRows / blockSize + !!(nRows % blockSize);

        SafeStatus safeStat;

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            size_t begin = iBlock * blockSize;
            size_t end   = services::internal::min<cpu, size_t>(begin + blockSize, nRows);

            ReadRows<FPType, cpu> dataRows(const_cast<NumericTable &>(ntData), begin, end - begin);
            DAAL_CHECK_BLOCK_STATUS_THR(dataRows);
            const FPType * const data = dataRows.get();

            FPType * r = res + begin;

            for (size_t i = 0; i < end - begin; i++)
            {
                FPType sum = FPType(0);
                PRAGMA_IVDEP
                PRAGMA_ICC_NO16(omp simd reduction(+ : sum))
                for (size_t j = 0; j < nCols; j++)
                {
                    sum += data[i * nCols + j] * data[i * nCols + j];
                }
                r[i] = sum;
            }
        });

        return safeStat.detach();
    }

    // compute (A x B')
    void computeABt(const FPType * const a, const FPType * const b, const size_t nRowsA, const size_t nColsA, const size_t nRowsB, FPType * const out)
    {
        const char transa    = 't';
        const char transb    = 'n';
        const DAAL_INT _m    = nRowsB;
        const DAAL_INT _n    = nRowsA;
        const DAAL_INT _k    = nColsA;
        const FPType alpha   = 1.0;
        const DAAL_INT lda   = nColsA;
        const DAAL_INT ldy   = nColsA;
        const FPType beta    = 0.0;
        const DAAL_INT ldaty = nRowsB;

        Blas<FPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, b, &lda, a, &ldy, &beta, out, &ldaty);
    }

    const NumericTable & _a;
    const NumericTable & _b;
    const bool _squared;

    TArray<FPType, cpu> normBufferA;
    TArray<FPType, cpu> normBufferB;
};

template <typename FPType, CpuType cpu>
class MinkowskiDistances : public PairwiseDistances<FPType, cpu>
{
public:
    MinkowskiDistances(const NumericTable & a, const NumericTable & b, const bool powered = true, const double p = 2.)
        : _a(a), _b(b), _powered(powered), _p(p)
    {}

    virtual ~MinkowskiDistances() {}

    virtual services::Status init()
    {
        services::Status s;
        return s;
    }

    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)
    {
        computeBatchImpl(a, b, aOffset, aSize, bOffset, bSize, res);

        return services::Status();
    }

protected:
    FPType computeDistance(const FPType * x, const FPType * y, const size_t n)
    {
        daal::internal::mkl::MklMath<FPType, cpu> math;

        FPType d = 0;

        if (_p == 1.0)
        {
            for (size_t i = 0; i < n; ++i)
            {
                d += math.sFabs(x[i] - y[i]);
            }

            return d;
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                d += math.sPowx(math.sFabs(x[i] - y[i]), _p);
            }

            if (!_powered) return math.sPowx(d, 1.0 / _p);

            return d;
        }
    }

    services::Status computeBatchImpl(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                      FPType * const res)
    {
        daal::internal::mkl::MklMath<FPType, cpu> math;

        const size_t nDims = _a.getNumberOfColumns();
        const size_t nX    = aSize;
        const size_t nY    = bSize;

        const FPType * x = a;
        DAAL_CHECK_MALLOC(x);

        const FPType * y = b;
        DAAL_CHECK_MALLOC(y);

        const size_t BlockSize        = 64;
        const size_t THREADING_BORDER = 32768;
        const size_t nBlocksX         = nX / BlockSize;
        const size_t nBlocksY         = nY / BlockSize;
        const size_t nThreads         = threader_get_threads_number();

        if (nThreads > 1 && nX * nY > THREADING_BORDER)
        {
            daal::threader_for(nBlocksX, nBlocksX, [&](size_t iBlockX) {
                const size_t startX = iBlockX * BlockSize;
                const size_t endX   = (nBlocksX - iBlockX - 1) ? startX + BlockSize : nX;

                daal::threader_for(nBlocksY, nBlocksY, [&](size_t iBlockY) {
                    const size_t startY = iBlockY * BlockSize;
                    const size_t endY   = (nBlocksY - iBlockY - 1) ? startY + BlockSize : nY;

                    for (size_t ix = startX; ix < endX; ++ix)
                        for (size_t iy = startY; iy < endY; ++iy)
                        {
                            res[ix * nY + iy] = computeDistance(x + ix * nDims, y + iy * nDims, nDims);
                        }
                });
            });
        }
        else
        {
            for (size_t ix = 0; ix < nX; ++ix)
                for (size_t iy = 0; iy < nY; ++iy)
                {
                    res[ix * nY + iy] = computeDistance(x + ix * nDims, y + iy * nDims, nDims);
                }
        }

        return services::Status();
    }

private:
    const NumericTable & _a;
    const NumericTable & _b;
    const double _p;
    const bool _powered;
};

template <typename FPType, CpuType cpu>
class ChebychevDistances : public PairwiseDistances<FPType, cpu>
{
public:
    ChebychevDistances(const NumericTable & a, const NumericTable & b) : _a(a), _b(b) {}

    virtual ~ChebychevDistances() {}

    virtual services::Status init()
    {
        services::Status s;
        return s;
    }

    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)
    {
        computeBatchImpl(a, b, aOffset, aSize, bOffset, bSize, res);

        return services::Status();
    }

protected:
    FPType computeDistance(const FPType * x, const FPType * y, const size_t n)
    {
        daal::internal::mkl::MklMath<FPType, cpu> math;

        FPType d = 0;

        for (size_t i = 0; i < n; ++i)
        {
            if (math.sFabs(x[i] - y[i]) > d)
            {
                d = math.sFabs(x[i] - y[i]);
            }
        }

        return d;
    }

    services::Status computeBatchImpl(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                      FPType * const res)
    {
        daal::internal::mkl::MklMath<FPType, cpu> math;

        const size_t nDims = _a.getNumberOfColumns();
        const size_t nX    = aSize;
        const size_t nY    = bSize;

        const FPType * x = a;
        DAAL_CHECK_MALLOC(x);

        const FPType * y = b;
        DAAL_CHECK_MALLOC(y);

        const size_t BlockSize        = 64;
        const size_t THREADING_BORDER = 32768;
        const size_t nBlocksX         = nX / BlockSize;
        const size_t nBlocksY         = nY / BlockSize;
        const size_t nThreads         = threader_get_threads_number();

        if (nThreads > 1 && nX * nY > THREADING_BORDER)
        {
            daal::threader_for(nBlocksX, nBlocksX, [&](size_t iBlockX) {
                const size_t startX = iBlockX * BlockSize;
                const size_t endX   = (nBlocksX - iBlockX - 1) ? startX + BlockSize : nX;

                daal::threader_for(nBlocksY, nBlocksY, [&](size_t iBlockY) {
                    const size_t startY = iBlockY * BlockSize;
                    const size_t endY   = (nBlocksY - iBlockY - 1) ? startY + BlockSize : nY;

                    for (size_t ix = startX; ix < endX; ++ix)
                        for (size_t iy = startY; iy < endY; ++iy)
                        {
                            res[ix * nY + iy] = computeDistance(x + ix * nDims, y + iy * nDims, nDims);
                        }
                });
            });
        }
        else
        {
            for (size_t ix = 0; ix < nX; ++ix)
                for (size_t iy = 0; iy < nY; ++iy)
                {
                    res[ix * nY + iy] = computeDistance(x + ix * nDims, y + iy * nDims, nDims);
                }
        }

        return services::Status();
    }

private:
    const NumericTable & _a;
    const NumericTable & _b;
};

#if defined(__INTEL_COMPILER)

template <>
float MinkowskiDistances<float, avx512>::computeDistance(const float * x, const float * y, const size_t n)
{
    daal::internal::mkl::MklMath<float, avx512> math;
    float * tmp = new float[16];
    float d     = 0.0;

    __m512 * ptr512x = (__m512 *)x;
    __m512 * ptr512y = (__m512 *)y;

    if (_p == 1.0)
    {
        for (size_t i = 0; i < n / 16; ++i)
        {
            d += _mm512_reduce_add_ps(_mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
        }

        for (size_t i = (n / 16) * 16; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;
    }
    else
    {
        for (size_t i = 0; i < n / 16; ++i)
        {
            _mm512_storeu_ps(tmp, _mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
            math.vPowx(4, tmp, _p, tmp);
            d += _mm512_reduce_add_ps(_mm512_loadu_ps(tmp));
        }

        delete[] tmp;

        for (size_t i = (n / 16) * 16; i < n; ++i)
        {
            d += math.sPowx(math.sFabs(x[i] - y[i]), _p);
        }

        if (!_powered) return math.sPowx(d, 1.0 / _p);

        return d;
    }
}

template <>
double MinkowskiDistances<double, avx512>::computeDistance(const double * x, const double * y, const size_t n)
{
    daal::internal::mkl::MklMath<double, avx512> math;

    double d          = 0.0;
    double * tmp      = new double[8];
    __m512d * ptr512x = (__m512d *)x;
    __m512d * ptr512y = (__m512d *)y;

    if (_p == 1.0)
    {
        for (size_t i = 0; i < n / 8; ++i)
        {
            d += _mm512_reduce_add_pd(_mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
        }

        for (size_t i = (n / 8) * 8; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;
    }
    else
    {
        for (size_t i = 0; i < n / 8; ++i)
        {
            _mm512_storeu_pd(tmp, _mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
            math.vPowx(8, tmp, _p, tmp);
            d += _mm512_reduce_add_pd(_mm512_loadu_pd(tmp));
        }

        delete[] tmp;

        for (size_t i = (n / 8) * 8; i < n; ++i)
        {
            d += math.sPowx(math.sFabs(x[i] - y[i]), _p);
        }

        if (!_powered) return math.sPowx(d, 1.0 / _p);

        return d;
    }
}

template <>
float ChebychevDistances<float, avx512>::computeDistance(const float * x, const float * y, const size_t n)
{
    daal::internal::mkl::MklMath<float, avx512> math;
    float * tmp = new float[16];
    float d     = 0.0;

    __m512 * ptr512x = (__m512 *)x;
    __m512 * ptr512y = (__m512 *)y;

    __m512 tmp512 = _mm512_abs_ps(_mm512_sub_ps(ptr512x[0], ptr512y[0]));

    for (size_t i = 1; i < n / 16; ++i)
    {
        tmp512 = _mm512_max_ps(tmp512, _mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
    }

    _mm512_storeu_ps(tmp, tmp512);

    for (size_t i = 0; i < 16; ++i)
    {
        if (tmp[i] > d)
        {
            d = tmp[i];
        }
    }

    delete[] tmp;

    for (size_t i = (n / 16) * 16; i < n; ++i)
    {
        if (math.sFabs(x[i] - y[i]) > d)
        {
            d = math.sFabs(x[i] - y[i]);
        }
    }

    return d;
}

template <>
double ChebychevDistances<double, avx512>::computeDistance(const double * x, const double * y, const size_t n)
{
    daal::internal::mkl::MklMath<double, avx512> math;

    double d          = 0.0;
    double * tmp      = new double[8];
    __m512d * ptr512x = (__m512d *)x;
    __m512d * ptr512y = (__m512d *)y;

    __m512d tmp512 = _mm512_abs_pd(_mm512_sub_pd(ptr512x[0], ptr512y[0]));

    for (size_t i = 1; i < n / 8; ++i)
    {
        tmp512 = _mm512_max_pd(tmp512, _mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
    }

    _mm512_storeu_pd(tmp, tmp512);

    for (size_t i = 0; i < 8; ++i)
    {
        if (tmp[i] > d)
        {
            d = tmp[i];
        }
    }

    delete[] tmp;

    for (size_t i = (n / 8) * 8; i < n; ++i)
    {
        if (math.sFabs(x[i] - y[i]) > d)
        {
            d = math.sFabs(x[i] - y[i]);
        }
    }

    return d;
}

#endif

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
