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
#include "src/algorithms/service_error_handling.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/features/defines.h"
#include "services/error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_allocators.h"
#include "src/services/service_arrays.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_dispatch.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "immintrin.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

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

enum class PairwiseDistanceType
{
    minkowski,
    euclidean,
    chebyshev,
    cosine,
};

template <typename FPType, CpuType cpu>
class PairwiseDistances
{
public:
    virtual ~PairwiseDistances() {};

    virtual services::Status init() = 0;

    virtual PairwiseDistanceType getType() = 0;

    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res) = 0;

    virtual services::Status finalize(const size_t n, FPType * a) = 0;
};

// compute: sum(A^2, 2) + sum(B^2, 2) -2*A*B'
template <typename FPType, CpuType cpu>
class EuclideanDistances : public PairwiseDistances<FPType, cpu>
{
public:
    EuclideanDistances(const NumericTable & a, const NumericTable & b, bool squared = true, bool isSqrtNorm = false)
        : _a(a), _b(b), _squared(squared), _isSqrtNorm(isSqrtNorm)
    {}

    virtual ~EuclideanDistances() DAAL_C11_OVERRIDE {}

    PairwiseDistanceType getType() DAAL_C11_OVERRIDE { return PairwiseDistanceType::euclidean; }

    services::Status init() DAAL_C11_OVERRIDE
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
    services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                  FPType * const res) DAAL_C11_OVERRIDE
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

    services::Status finalize(const size_t n, FPType * a) DAAL_C11_OVERRIDE
    {
        const size_t blockSize = 512;
        const size_t nBlocks   = n / blockSize + !!(n % blockSize);

        SafeStatus safeStat;

        for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
        {
            const size_t begin = iBlock * blockSize;
            const size_t end   = services::internal::min<cpu, size_t>(begin + blockSize, n);
            const size_t count = end - begin;

            // max(0, d) to remove negative distances before Sqrt
            for (size_t i = begin; i < end; ++i)
            {
                a[i] = services::internal::max<cpu, FPType>(FPType(0), a[i]);
            }
            Math<FPType, cpu>::vSqrt(count, a + begin, a + begin);
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
                if (_isSqrtNorm)
                {
                    Math<FPType, cpu>::vSqrt(end - begin, r, r);
                }
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
    const bool _isSqrtNorm;

    TArray<FPType, cpu> normBufferA;
    TArray<FPType, cpu> normBufferB;
};

// compute: A*B' / (sum(A^2, 2) * sum(B^2, 2))
template <typename FPType, CpuType cpu>
class CosineDistances : public EuclideanDistances<FPType, cpu>
{
private:
    using super = EuclideanDistances<FPType, cpu>;

public:
    CosineDistances(const NumericTable & a, const NumericTable & b) : super(a, b, true, true) {}

    virtual ~CosineDistances() DAAL_C11_OVERRIDE {}

    PairwiseDistanceType getType() DAAL_C11_OVERRIDE { return PairwiseDistanceType::cosine; }

    services::Status finalize(const size_t n, FPType * a) DAAL_C11_OVERRIDE { return services::Status(); }

    // output:  Row-major matrix of size { aSize x bSize }
    services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                  FPType * const res) DAAL_C11_OVERRIDE
    {
        const size_t nRowsA = aSize;
        const size_t nColsA = super::_a.getNumberOfColumns();
        const size_t nRowsB = bSize;

        const size_t nRowsC = nRowsA;
        const size_t nColsC = nRowsB;

        super::computeABt(a, b, nRowsA, nColsA, nRowsB, res);

        const FPType * const aa = super::normBufferA.get() + aOffset;
        const FPType * const bb = (&(super::_a) == &(super::_b)) ? super::normBufferA.get() + bOffset : super::normBufferB.get() + bOffset;

        for (size_t i = 0; i < nRowsC; i++)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nColsC; j++)
            {
                res[i * nColsC + j] = FPType(1) - (res[i * nColsC + j] / (aa[i] * bb[j]));
            }
        }
        return services::Status();
    }
};

template <typename FPType, CpuType cpu>
class MinkowskiDistances : public PairwiseDistances<FPType, cpu>
{
public:
    MinkowskiDistances(const NumericTable & a, const NumericTable & b, const bool powered = true, const double p = 2.0)
        : _a(a), _b(b), _powered(powered), _p(p)
    {}

    virtual ~MinkowskiDistances() DAAL_C11_OVERRIDE {}

    PairwiseDistanceType getType() DAAL_C11_OVERRIDE { return PairwiseDistanceType::minkowski; }

    services::Status init() DAAL_C11_OVERRIDE
    {
        services::Status s;
        return s;
    }

    services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                  FPType * const res) DAAL_C11_OVERRIDE
    {
        computeBatchImpl(a, b, aOffset, aSize, bOffset, bSize, res);

        return services::Status();
    }

    services::Status finalize(const size_t n, FPType * a) DAAL_C11_OVERRIDE
    {
        if (_p != 1.0)
        {
            daal::internal::mkl::MklMath<FPType, cpu> math;
            math.vPowx(n, a, 1.0 / _p, a);
        }
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

        for (size_t ix = 0; ix < nX; ++ix)
        {
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
class ChebyshevDistances : public PairwiseDistances<FPType, cpu>
{
public:
    ChebyshevDistances(const NumericTable & a, const NumericTable & b) : _a(a), _b(b) {}

    virtual ~ChebyshevDistances() DAAL_C11_OVERRIDE {}

    PairwiseDistanceType getType() DAAL_C11_OVERRIDE { return PairwiseDistanceType::chebyshev; }

    services::Status init() DAAL_C11_OVERRIDE
    {
        services::Status s;
        return s;
    }

    services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                  FPType * const res) DAAL_C11_OVERRIDE
    {
        computeBatchImpl(a, b, aOffset, aSize, bOffset, bSize, res);

        return services::Status();
    }

    services::Status finalize(const size_t n, FPType * a) DAAL_C11_OVERRIDE { return services::Status(); }

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

        for (size_t ix = 0; ix < nX; ++ix)
        {
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

    const size_t vecSize   = 16;
    float d                = 0.0;
    float * tmp            = new float[vecSize];
    const __m512 * ptr512x = (__m512 *)x;
    const __m512 * ptr512y = (__m512 *)y;

    if (_p == 1.0)
    {
        size_t i = 0;
        for (; vecSize * i < n; ++i)
        {
            d += _mm512_reduce_add_ps(_mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
        }

        for (i *= vecSize; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;
    }
    else
    {
        size_t i = 0;
        for (; vecSize * i < n; ++i)
        {
            _mm512_storeu_ps(tmp, _mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
            math.vPowx(4, tmp, _p, tmp);
            d += _mm512_reduce_add_ps(_mm512_loadu_ps(tmp));
        }

        delete[] tmp;

        for (i *= vecSize; i < n; ++i)
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

    const size_t vecSize    = 8;
    double d                = 0.0;
    double * tmp            = new double[vecSize];
    const __m512d * ptr512x = (__m512d *)x;
    const __m512d * ptr512y = (__m512d *)y;

    if (_p == 1.0)
    {
        size_t i = 0;
        for (; vecSize * i < n; ++i)
        {
            d += _mm512_reduce_add_pd(_mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
        }

        for (i *= vecSize; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;
    }
    else
    {
        size_t i = 0;
        for (; vecSize * i < n; ++i)
        {
            _mm512_storeu_pd(tmp, _mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
            math.vPowx(8, tmp, _p, tmp);
            d += _mm512_reduce_add_pd(_mm512_loadu_pd(tmp));
        }

        delete[] tmp;

        for (i *= vecSize; i < n; ++i)
        {
            d += math.sPowx(math.sFabs(x[i] - y[i]), _p);
        }

        if (!_powered) return math.sPowx(d, 1.0 / _p);

        return d;
    }
}

template <>
float ChebyshevDistances<float, avx512>::computeDistance(const float * x, const float * y, const size_t n)
{
    daal::internal::mkl::MklMath<float, avx512> math;

    const size_t vecSize   = 16;
    float d                = 0.0;
    float * tmp            = new float[vecSize];
    const __m512 * ptr512x = (__m512 *)x;
    const __m512 * ptr512y = (__m512 *)y;

    __m512 tmp512 = _mm512_abs_ps(_mm512_sub_ps(ptr512x[0], ptr512y[0]));

    size_t i = 1;
    for (; vecSize * i < n; ++i)
    {
        tmp512 = _mm512_max_ps(tmp512, _mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
    }

    _mm512_storeu_ps(tmp, tmp512);
    size_t ii = i;

    for (i = 0; i < vecSize; ++i)
    {
        d = (tmp[i] > d) ? tmp[i] : d;
    }

    delete[] tmp;

    for (ii *= vecSize; ii < n; ++ii)
    {
        d = (math.sFabs(x[i] - y[i]) > d) ? math.sFabs(x[i] - y[i]) : d;
    }

    return d;
}

template <>
double ChebyshevDistances<double, avx512>::computeDistance(const double * x, const double * y, const size_t n)
{
    daal::internal::mkl::MklMath<double, avx512> math;

    const size_t vecSize    = 8;
    double d                = 0.0;
    double * tmp            = new double[8];
    const __m512d * ptr512x = (__m512d *)x;
    const __m512d * ptr512y = (__m512d *)y;

    __m512d tmp512 = _mm512_abs_pd(_mm512_sub_pd(ptr512x[0], ptr512y[0]));

    size_t i = 1;
    for (; vecSize * i < n; ++i)
    {
        tmp512 = _mm512_max_pd(tmp512, _mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
    }

    _mm512_storeu_pd(tmp, tmp512);
    size_t ii = i;

    for (i = 0; i < vecSize; ++i)
    {
        d = (tmp[i] > d) ? tmp[i] : d;
    }

    delete[] tmp;

    for (ii *= vecSize; ii < n; ++ii)
    {
        d = (math.sFabs(x[i] - y[i]) > d) ? math.sFabs(x[i] - y[i]) : d;
    }

    return d;
}

#endif

template <typename FPType, CpuType cpu>
bool solveEquationsSystemWithCholesky(FPType * a, FPType * b, size_t n, size_t nX, bool sequential)
{
    /* POTRF and POTRS parameters */
    char uplo     = 'U';
    DAAL_INT info = 0;

    /* Perform L*L' factorization of A */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxpotrf(&uplo, (DAAL_INT *)&n, a, (DAAL_INT *)&n, &info);
    }
    else
    {
        Lapack<FPType, cpu>::xpotrf(&uplo, (DAAL_INT *)&n, a, (DAAL_INT *)&n, &info);
    }
    if (info != 0) return false;

    /* Solve L*L' * x = b */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxpotrs(&uplo, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, b, (DAAL_INT *)&n, &info);
    }
    else
    {
        Lapack<FPType, cpu>::xpotrs(&uplo, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, b, (DAAL_INT *)&n, &info);
    }
    return (info == 0);
}

template <typename FPType, CpuType cpu>
bool solveEquationsSystemWithPLU(FPType * a, FPType * b, size_t n, size_t nX, bool sequential, bool extendFromSymmetric)
{
    if (extendFromSymmetric)
    {
        /* Extend symmetric matrix to generic through filling of upper triangle */
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < i; ++j)
            {
                a[j * n + i] = a[i * n + j];
            }
        }
    }

    /* GETRF and GETRS parameters */
    char trans    = 'N';
    DAAL_INT info = 0;

    TArrayScalable<DAAL_INT, cpu> ipiv(n);
    DAAL_CHECK_MALLOC(ipiv.get());

    /* Perform P*L*U factorization of A */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxgetrf((DAAL_INT *)&n, (DAAL_INT *)&n, a, (DAAL_INT *)&n, ipiv.get(), &info);
    }
    else
    {
        Lapack<FPType, cpu>::xgetrf((DAAL_INT *)&n, (DAAL_INT *)&n, a, (DAAL_INT *)&n, ipiv.get(), &info);
    }
    if (info != 0) return false;

    /* Solve P*L*U * x = b */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxgetrs(&trans, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, ipiv.get(), b, (DAAL_INT *)&n, &info);
    }
    else
    {
        Lapack<FPType, cpu>::xgetrs(&trans, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, ipiv.get(), b, (DAAL_INT *)&n, &info);
    }
    return (info == 0);
}

template <typename FPType, CpuType cpu>
bool solveSymmetricEquationsSystem(FPType * a, FPType * b, size_t n, size_t nX, bool sequential)
{
    /* Copy data for fallback from Cholesky to PLU factorization */
    TArrayScalable<FPType, cpu> aCopy(n * n);
    TArrayScalable<FPType, cpu> bCopy(n);
    DAAL_CHECK_MALLOC(aCopy.get());
    DAAL_CHECK_MALLOC(bCopy.get());

    int copy_status = services::internal::daal_memcpy_s(aCopy.get(), n * n * sizeof(FPType), a, n * n * sizeof(FPType));
    copy_status += services::internal::daal_memcpy_s(bCopy.get(), n * sizeof(FPType), b, n * sizeof(FPType));

    if (copy_status != 0) return false;

    /* Try to solve with Cholesky factorization */
    if (!solveEquationsSystemWithCholesky<FPType, cpu>(a, b, n, nX, sequential))
    {
        /* Fallback to PLU factorization */
        bool status = solveEquationsSystemWithPLU<FPType, cpu>(aCopy.get(), bCopy.get(), n, nX, sequential, true);
        if (status)
        {
            status = status && (services::internal::daal_memcpy_s(b, n * sizeof(FPType), bCopy.get(), n * sizeof(FPType)) == 0);
        }
        return status;
    }
    return true;
}
} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
