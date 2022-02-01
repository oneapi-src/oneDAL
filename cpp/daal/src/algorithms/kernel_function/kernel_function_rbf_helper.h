/* file: kernel_function_rbf_helper.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __KERNEL_FUNCTION_RBF_HELPER_H__
#define __KERNEL_FUNCTION_RBF_HELPER_H__

#include "src/externals/service_math.h"

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
class HelperKernelRBF
{
public:
    static services::Status postGemmPart(algorithmFPType * const mklBuff, const algorithmFPType * const sqrA1i, const algorithmFPType sqrA2i,
                                         const algorithmFPType coeff, const algorithmFPType expExpThreshold, const size_t n,
                                         algorithmFPType * const dataRBlock);
};

template <typename algorithmFPType, CpuType cpu>
services::Status HelperKernelRBF<algorithmFPType, cpu>::postGemmPart(algorithmFPType * const mklBuff, const algorithmFPType * const sqrA1i,
                                                                     const algorithmFPType sqrA2i, const algorithmFPType coeff,
                                                                     const algorithmFPType expExpThreshold, const size_t n,
                                                                     algorithmFPType * const dataRBlock)
{
    const algorithmFPType negTwo = algorithmFPType(-2.0);
    for (size_t i = 0; i < n; ++i)
    {
        const algorithmFPType rbf = (mklBuff[i] * negTwo + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]                = rbf > expExpThreshold ? rbf : expExpThreshold;
    }
    Math<algorithmFPType, cpu>::vExp(n, mklBuff, dataRBlock);
    return services::Status();
}

#if defined(__INTEL_COMPILER)

template <>
services::Status HelperKernelRBF<double, avx512>::postGemmPart(double * const mklBuff, const double * const sqrA1i, const double sqrA2i,
                                                               const double coeff, const double expExpThreshold, const size_t n,
                                                               double * const dataRBlock)
{
    const double negTwo              = -2.0;
    const __m512d sqrA2iVec          = _mm512_set1_pd(sqrA2i);
    const __m512d coeffVec           = _mm512_set1_pd(coeff);
    const __m512d negTwoVec          = _mm512_set1_pd(negTwo);
    const __m512d expExpThresholdVec = _mm512_set1_pd(expExpThreshold);

    size_t i = 0;
    for (; (i + 8) < n; i += 8)
    {
        const __m512d sqrDataA1Vec = _mm512_load_pd(&sqrA1i[i]);
        __m512d sqrDataA1CoeffVec  = _mm512_mul_pd(sqrDataA1Vec, coeffVec);
        __m512d mklBuffVec         = _mm512_load_pd(&mklBuff[i]);
        mklBuffVec                 = _mm512_mul_pd(mklBuffVec, negTwoVec);
        __m512d rbfVec             = _mm512_add_pd(mklBuffVec, sqrA2iVec);
        rbfVec                     = _mm512_fmadd_pd(rbfVec, coeffVec, sqrDataA1CoeffVec);
        rbfVec                     = _mm512_max_pd(rbfVec, expExpThresholdVec);

        _mm512_store_pd(&mklBuff[i], rbfVec);
    }
    for (; i < n; i++)
    {
        const double rbf = (mklBuff[i] * negTwo + sqrA2i + sqrA1i[i]) * coeff;
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
services::Status HelperKernelRBF<float, avx512>::postGemmPart(float * const mklBuff, const float * const sqrA1i, const float sqrA2i,
                                                              const float coeff, const float expExpThreshold, const size_t n,
                                                              float * const dataRBlock)
{
    const float negTwo              = -2.0f;
    const __m512 sqrA2iVec          = _mm512_set1_ps(sqrA2i);
    const __m512 coeffVec           = _mm512_set1_ps(coeff);
    const __m512 negTwoVec          = _mm512_set1_ps(negTwo);
    const __m512 expExpThresholdVec = _mm512_set1_ps(expExpThreshold);

    size_t i = 0;

    for (; (i + 16) < n; i += 16)
    {
        const __m512 sqrDataA1Vec = _mm512_load_ps(&sqrA1i[i]);
        __m512 sqrDataA1CoeffVec  = _mm512_mul_ps(sqrDataA1Vec, coeffVec);
        __m512 mklBuffVec         = _mm512_load_ps(&mklBuff[i]);
        mklBuffVec                = _mm512_mul_ps(mklBuffVec, negTwoVec);
        __m512 rbfVec             = _mm512_add_ps(mklBuffVec, sqrA2iVec);
        rbfVec                    = _mm512_fmadd_ps(rbfVec, coeffVec, sqrDataA1CoeffVec);
        rbfVec                    = _mm512_max_ps(rbfVec, expExpThresholdVec);
        _mm512_store_ps(&mklBuff[i], rbfVec);
    }
    for (; i < n; i++)
    {
        const float rbf = (mklBuff[i] * negTwo + sqrA2i + sqrA1i[i]) * coeff;
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

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
