/* file: kernel_function_rbf_dense_default_impl.i */
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
//  RBF kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_ittnotify.h"
#include "src/threading/threading.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct KernelRBFTask
{
public:
    DAAL_NEW_DELETE();
    algorithmFPType * mklBuff;
    algorithmFPType * sqrDataA1;
    algorithmFPType * sqrDataA2;

    static KernelRBFTask * create(const size_t blockSize, const bool isEqualMatrix)
    {
        auto object = new KernelRBFTask(blockSize, isEqualMatrix);
        if (object && object->isValid()) return object;
        delete object;
        return nullptr;
    }

    bool isValid() const { return _buff.get(); }

private:
    KernelRBFTask(const size_t blockSize, const bool isEqualMatrix)
    {
        const size_t buffASize = isEqualMatrix ? blockSize : 2 * blockSize;
        _buff.reset(blockSize * blockSize + buffASize);

        mklBuff   = &_buff[0];
        sqrDataA1 = &_buff[blockSize * blockSize];
        sqrDataA2 = isEqualMatrix ? sqrDataA1 : &sqrDataA1[blockSize];
    }

    TArrayScalable<algorithmFPType, cpu> _buff;
};

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nFeatures = a1->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.get();

    ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.get();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);

    //compute
    const Parameter * rbfPar          = static_cast<const Parameter *>(par);
    const algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    algorithmFPType factor            = 0.0;
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        algorithmFPType diff = (dataA1[i] - dataA2[i]);
        factor += diff * diff;
    }
    factor *= -0.5 * invSqrSigma;
    daal::internal::Math<algorithmFPType, cpu>::vExp(1, &factor, mtR.get());
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.get();

    ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.get();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const Parameter * rbfPar          = static_cast<const Parameter *>(par);
    const algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    for (size_t i = 0; i < nVectors1; i++)
    {
        algorithmFPType factor = 0.0;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            algorithmFPType diff = (dataA1[i * nFeatures + j] - dataA2[j]);
            factor += diff * diff;
        }
        dataR[i] = -0.5 * invSqrSigma * factor;

        if (dataR[i] < Math<algorithmFPType, cpu>::vExpThreshold())
        {
            dataR[i] = Math<algorithmFPType, cpu>::vExpThreshold();
        }
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(nVectors1, dataR, dataR);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::postGemmPart(algorithmFPType * const mklBuff,
                                                                                 const algorithmFPType * const sqrA1i, const algorithmFPType sqrA2i,
                                                                                 const algorithmFPType coeff, const algorithmFPType expExpThreshold,
                                                                                 const size_t n, algorithmFPType * const dataRBlock)
{
    for (size_t i = 0; i < n; ++i)
    {
        const algorithmFPType rbf = (mklBuff[i] + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]                = rbf > expExpThreshold ? rbf : expExpThreshold;
    }
    Math<algorithmFPType, cpu>::vExp(n, mklBuff, dataRBlock);
    return services::Status();
}

#if defined(__INTEL_COMPILER)

template <>
services::Status KernelImplRBF<defaultDense, double, avx512>::postGemmPart(double * const mklBuff, const double * const sqrA1i, const double sqrA2i,
                                                                           const double coeff, const double expExpThreshold, const size_t n,
                                                                           double * const dataRBlock)
{
    const __m512d sqrA2iVec          = _mm512_set1_pd(sqrA2i);
    const __m512d coeffVec           = _mm512_set1_pd(coeff);
    const __m512d expExpThresholdVec = _mm512_set1_pd(expExpThreshold);

    size_t i = 0;
    for (; (i + 8) < n; i += 8)
    {
        const __m512d sqrDataA1Vec = _mm512_load_pd(&sqrA1i[i]);
        __m512d sqrDataA1CoeffVec  = _mm512_mul_pd(sqrDataA1Vec, coeffVec);
        const __m512d mklBuffVec   = _mm512_load_pd(&mklBuff[i]);
        __m512d rbfVec             = _mm512_add_pd(mklBuffVec, sqrA2iVec);
        rbfVec                     = _mm512_fmadd_pd(rbfVec, coeffVec, sqrDataA1CoeffVec);
        rbfVec                     = _mm512_max_pd(rbfVec, expExpThresholdVec);

        _mm512_store_pd(&mklBuff[i], rbfVec);
    }
    for (; i < n; i++)
    {
        const double rbf = (mklBuff[i] + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]       = rbf > expExpThreshold ? rbf : expExpThreshold;
    }

    Math<double, avx512>::vExp(n, mklBuff, mklBuff);
    i = 0;

    const size_t align = ((64 - (reinterpret_cast<size_t>(dataRBlock) & 63)) & 63) >> 3;
    for (; i < align; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    for (; (i + 8) < n; i += 8)
    {
        const __m512d mklBuffVec = _mm512_loadu_pd(&mklBuff[i]);
        _mm512_stream_pd(&dataRBlock[i], mklBuffVec);
    }
    for (; i < n; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    return services::Status();
}

template <>
services::Status KernelImplRBF<defaultDense, float, avx512>::postGemmPart(float * const mklBuff, const float * const sqrA1i, const float sqrA2i,
                                                                          const float coeff, const float expExpThreshold, const size_t n,
                                                                          float * const dataRBlock)
{
    const __m512 sqrA2iVec          = _mm512_set1_ps(sqrA2i);
    const __m512 coeffVec           = _mm512_set1_ps(coeff);
    const __m512 expExpThresholdVec = _mm512_set1_ps(expExpThreshold);

    size_t i = 0;

    for (; (i + 16) < n; i += 16)
    {
        const __m512 sqrDataA1Vec = _mm512_load_ps(&sqrA1i[i]);
        __m512 sqrDataA1CoeffVec  = _mm512_mul_ps(sqrDataA1Vec, coeffVec);
        const __m512 mklBuffVec   = _mm512_load_ps(&mklBuff[i]);
        __m512 rbfVec             = _mm512_add_ps(mklBuffVec, sqrA2iVec);
        rbfVec                    = _mm512_fmadd_ps(rbfVec, coeffVec, sqrDataA1CoeffVec);
        rbfVec                    = _mm512_max_ps(rbfVec, expExpThresholdVec);
        _mm512_store_ps(&mklBuff[i], rbfVec);
    }
    for (; i < n; i++)
    {
        const float rbf = (mklBuff[i] + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]      = rbf > expExpThreshold ? rbf : expExpThreshold;
    }

    Math<float, avx512>::vExp(n, mklBuff, mklBuff);
    i = 0;

    const size_t align = ((64 - (reinterpret_cast<size_t>(dataRBlock) & 63)) & 63) >> 2;
    for (; i < align; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    for (; (i + 16) < n; i += 16)
    {
        const __m512 mklBuffVec = _mm512_loadu_ps(&mklBuff[i]);
        _mm512_stream_ps(&dataRBlock[i], mklBuffVec);
    }
    for (; i < n; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    return services::Status();
}

#endif

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const ParameterBase * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.MatrixMatrix);

    SafeStatus safeStat;

    const size_t nVectors1   = a1->getNumberOfRows();
    const size_t nVectors2   = a2->getNumberOfRows();
    const size_t nFeatures   = a1->getNumberOfColumns();
    const bool isEqualMatrix = a1 == a2;

    const Parameter * rbfPar    = static_cast<const Parameter *>(par);
    const algorithmFPType coeff = (algorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));

    char trans = 'T', notrans = 'N';
    DAAL_INT one         = 1;
    algorithmFPType zero = 0.0, negTwo = -2.0;

    const bool isSOARes = r->getDataLayout() & NumericTableIface::soa;

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nVectors1, nVectors2);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors1 + nVectors2, sizeof(algorithmFPType));

    const size_t blockSize                = 256;
    const size_t nBlocks1                 = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2                 = nVectors2 / blockSize + !!(nVectors2 % blockSize);
    const algorithmFPType expExpThreshold = Math<algorithmFPType, cpu>::vExpThreshold();

    daal::tls<KernelRBFTask<algorithmFPType, cpu> *> tslTask([=, &safeStat]() {
        auto tlsData = KernelRBFTask<algorithmFPType, cpu>::create(blockSize, isEqualMatrix);
        if (!tlsData)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return tlsData;
    });

    daal::threader_for(nBlocks1, nBlocks1, [&](const size_t iBlock1) {
        DAAL_INT nRowsInBlock1 = (iBlock1 != nBlocks1 - 1) ? blockSize : nVectors1 - iBlock1 * blockSize;
        DAAL_INT startRow1     = iBlock1 * blockSize;

        ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), startRow1, nRowsInBlock1);
        DAAL_CHECK_BLOCK_STATUS_THR(mtA1);
        const algorithmFPType * const dataA1 = const_cast<algorithmFPType *>(mtA1.get());

        WriteOnlyRows<algorithmFPType, cpu> mtRRows;
        if (!isSOARes)
        {
            mtRRows.set(r, startRow1, nRowsInBlock1);
            DAAL_CHECK_MALLOC_THR(mtRRows.get());
        }
        daal::threader_for(nBlocks2, nBlocks2, [&, nVectors2, nBlocks2](const size_t iBlock2) {
            DAAL_INT nRowsInBlock2 = (iBlock2 != nBlocks2 - 1) ? blockSize : nVectors2 - iBlock2 * blockSize;
            DAAL_INT startRow2     = iBlock2 * blockSize;

            KernelRBFTask<algorithmFPType, cpu> * const tlsLocal = tslTask.local();

            algorithmFPType * const mklBuff   = tlsLocal->mklBuff;
            algorithmFPType * const sqrDataA1 = tlsLocal->sqrDataA1;
            algorithmFPType * const sqrDataA2 = tlsLocal->sqrDataA2;

            if (!isEqualMatrix)
            {
                for (size_t i = 0; i < nRowsInBlock1; ++i)
                {
                    const algorithmFPType * dataA1i = dataA1 + i * nFeatures;
                    sqrDataA1[i]                    = Blas<algorithmFPType, cpu>::xxdot((DAAL_INT *)&nFeatures, dataA1i, &one, dataA1i, &one);
                }
            }

            ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), startRow2, nRowsInBlock2);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * const dataA2 = const_cast<algorithmFPType *>(mtA2.get());

            for (size_t i = 0; i < nRowsInBlock2; ++i)
            {
                const algorithmFPType * dataA2i = dataA2 + i * nFeatures;
                sqrDataA2[i]                    = Blas<algorithmFPType, cpu>::xxdot((DAAL_INT *)&nFeatures, dataA2i, &one, dataA2i, &one);
            }

            DAAL_INT lda = nFeatures;
            DAAL_INT ldb = nFeatures;
            DAAL_INT ldc = blockSize;
            if (!isSOARes)
            {
                Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock2, &nRowsInBlock1, (DAAL_INT *)&nFeatures, &negTwo, dataA2, &ldb,
                                                   dataA1, &lda, &zero, mklBuff, &ldc);

                algorithmFPType * const dataR = mtRRows.get();
                for (size_t i = 0; i < nRowsInBlock1; ++i)
                {
                    const algorithmFPType sqrA1i         = sqrDataA1[i];
                    algorithmFPType * const dataRBlock   = &dataR[i * nVectors2 + startRow2];
                    algorithmFPType * const mklBuffBlock = &mklBuff[i * blockSize];
                    postGemmPart(mklBuffBlock, sqrDataA2, sqrA1i, coeff, expExpThreshold, nRowsInBlock2, dataRBlock);
                }
            }
            else
            {
                Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock1, &nRowsInBlock2, (DAAL_INT *)&nFeatures, &negTwo, dataA1, &lda,
                                                   dataA2, &ldb, &zero, mklBuff, &ldc);

                for (size_t j = 0; j < nRowsInBlock2; ++j)
                {
                    const algorithmFPType sqrA2i = sqrDataA2[j];
                    WriteOnlyColumns<algorithmFPType, cpu> mtRColumns(r, startRow2 + j, startRow1, nRowsInBlock1);
                    DAAL_CHECK_BLOCK_STATUS_THR(mtRColumns);
                    algorithmFPType * const dataRBlock   = mtRColumns.get();
                    algorithmFPType * const mklBuffBlock = &mklBuff[j * blockSize];
                    postGemmPart(mklBuffBlock, sqrDataA1, sqrA2i, coeff, expExpThreshold, nRowsInBlock1, dataRBlock);
                }
            }
        });
    });

    tslTask.reduce([](KernelRBFTask<algorithmFPType, cpu> * tlsLocal) { delete tlsLocal; });

    return services::Status();
}

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
