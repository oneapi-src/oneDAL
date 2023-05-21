/* file: svm_train_common_avx512_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * Contains optimizations for AVX512.
*/

#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <>
inline void HelperTrainSVM<float, avx512>::WSSjLocal(const size_t jStart, const size_t jEnd, const float * KiBlock, const float * kernelDiag,
                                                     const float * grad, const char * I, const float GMin, const float Kii, const float tau, int & Bj,
                                                     float & GMax, float & GMax2, float & delta, SignNuType signNuType)
{
    float fpMax      = MaxVal<float>::get();
    float GMax2Local = -fpMax; // store min(grad[i]) or max(y[i]*grad[i]), y[i]*grad[i] = -GMin2
    float GMaxLocal  = -fpMax; // store min(-b^2/a) or max(b^2/a), b^2/a = -GMin
    float GMinLocal  = GMin;

    float zero(0.0);
    float two(2.0);

    const char sign = getSign(signNuType);

    // generally, we find max(grad[i]).
    // for performance considerations, it is better to find max(grad[i]) instead.
    __m512 valGMax2 = _mm512_set1_ps(GMax2Local);

    // generally, we find max(b^2/a).
    __m512 valGMax = _mm512_set1_ps(GMaxLocal);

    // constant used to select j so that ygrad = grad[i] < GMax.
    // for performance considerations, we select j so that grad = grad[i] >= GMax.
    __m512 valGMin = _mm512_set1_ps(GMinLocal);

    // we minimize over index Bj. It is stored as vector.
    // We find 16 minimums in parallel to choose 1 later on reduction step.
    __m512i Bj_vec = _mm512_set1_epi32((int)-1);

    // some constants used during optimization
    // enum SVMVectorStatus low = 0x2
    __m128i vecSignLow =
        (signNuType == SignNuType::none) ? _mm_set1_epi8(low) : _mm_or_si128(_mm_set1_epi8(low), _mm_set1_epi8(sign)); // vector of masks
    __m512 two_vec = _mm512_set1_ps(two);                                                                              // vector of 2's
    __m512 Kii_vec = _mm512_set1_ps(Kii);                                                                              // vector of Kii's
    __m512 tau_vec = _mm512_set1_ps(tau);                                                                              // vector of tau's

    // mask_1: condition _I[j] is in I_low
    // mask_2: condition grad = grad[i] < (GE) GMin.
    // mask_3: condition a > tau

    size_t j_cur;
    for (j_cur = jStart; (j_cur + 16) <= jEnd; j_cur += 16)
    {
        // if (jEnd > jCur)  make mask?
        DAAL_ASSERT(j_cur <= services::internal::MaxVal<int>::get())
        __m128i vec_I = _mm_loadu_si128((__m128i *)(&I[j_cur]));

        // if (!(I[j] & sign) || (_I[j] & low) != low) { continue; }
        __mmask16 mask1_ = _mm_cmpeq_epi8_mask(_mm_and_si128(vec_I, vecSignLow), vecSignLow);

        __m512i Bj_vec_cur = _mm512_set1_epi32((int)(j_cur));
        // vector grad[jcur:jcur+16] = _grad[jcur:jcur+16]
        __m512 valGrad = _mm512_load_ps(&grad[j_cur]);

        // find min(grad[j]) instead - here's where we find vector of min's.
        // result can be updated only if _I[j] is in I_low.
        valGMax2 = _mm512_mask_max_ps(valGMax2, mask1_, valGrad, valGMax2);

        // we select j so that grad = grad[i] < (GE) GMin.
        __mmask16 mask2_ = _mm512_cmp_ps_mask(valGrad, valGMin, _CMP_GT_OS); // if (grad < GMaxLocal) { continue; }
        mask2_           = _mm512_kand(mask1_, mask2_);                      // combine with previous mask
        __m512 b_vec     = _mm512_sub_ps(valGMin, valGrad);                  // b = Gmin - grad

        // float a = Kii + _kernelDiag[j] - two * KiBlock[j - jStart] = Kii + _kernelDiag[j] - KiBlock[j - jStart] - KiBlock[j - jStart];
        // a_tmp = two * KiBlock[j - jStart] - kernelDiag[j]
        __m512 KiBlock_vec    = _mm512_loadu_ps(KiBlock + j_cur - jStart);
        __m512 kernelDiag_vec = _mm512_load_ps(&kernelDiag[j_cur]);
        __m512 a_vec          = _mm512_fmsub_ps(two_vec, KiBlock_vec, kernelDiag_vec);

        // originally, if a < 0, a = tau
        // mask3_ : 1 if Kii > a_tmp
        // if mask3_ : 1, a = Kii - a_tmp, else a = tau.
        __mmask16 mask3_ = _mm512_cmp_ps_mask(Kii_vec, a_vec, _CMP_GT_OS);
        a_vec            = _mm512_mask_sub_ps(tau_vec, mask3_, Kii_vec, a_vec);

        __m512 objFunc_vec = _mm512_div_ps(b_vec, a_vec);       // b/a = delta.
        objFunc_vec        = _mm512_mul_ps(objFunc_vec, b_vec); // we have objFunc = b * b / a

        // generally, objFunc = b^2/a.
        // if (objFunc > GMax)
        __mmask16 mask4_ = _mm512_cmp_ps_mask(objFunc_vec, valGMax, _CMP_GE_OS);
        mask4_           = _mm512_kand(mask2_, mask4_);                       // combine with previous masks
        valGMax          = _mm512_mask_mov_ps(valGMax, mask4_, objFunc_vec);  // GMax = b^2/a
        Bj_vec           = _mm512_mask_mov_epi32(Bj_vec, mask4_, Bj_vec_cur); // result index
    }

    // reduction:
    const float * const GMaxVal      = reinterpret_cast<float * const>(&valGMax);
    const int * const BjValOverall   = reinterpret_cast<int * const>(&Bj_vec);
    const float * const GMax2Overall = reinterpret_cast<float * const>(&valGMax2);

    GMax  = -fpMax;
    GMax2 = -fpMax;

    for (size_t k = 0; k < 16; ++k)
    {
        if (GMaxVal[k] > GMax)
        {
            GMax = GMaxVal[k];
            Bj   = BjValOverall[k] + k; // from partial index to final
        }
        if (GMax2Overall[k] > GMax2)
        {
            GMax2 = GMax2Overall[k];
        }
    }

    if (Bj != -1)
    {
        const float gradBj = grad[Bj];
        const float b      = GMin - gradBj;
        float a            = Kii + kernelDiag[Bj] - two * KiBlock[Bj - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        delta = b / a;
        GMax  = b * delta;
    }
    else
    {
        GMax  = -fpMax;
        GMax2 = -fpMax;
    }
    delta = -delta;
    if (j_cur < jEnd) // we have tail
    {
        int Bj_local     = -1;
        float GMax_local = -fpMax, GMax2_local = -fpMax, delta_local;
        WSSjLocalBaseline(j_cur, jEnd, (KiBlock + j_cur - jStart), kernelDiag, grad, I, GMin, Kii, tau, Bj_local, GMax_local, GMax2_local,
                          delta_local, signNuType);
        if (GMax_local > GMax)
        {
            GMax  = GMax_local;
            Bj    = Bj_local;
            delta = delta_local;
        }
        if (GMax2_local > GMax2)
        {
            GMax2 = GMax2_local;
        }
    }
}

template <>
inline void HelperTrainSVM<double, avx512>::WSSjLocal(const size_t jStart, const size_t jEnd, const double * KiBlock, const double * kernelDiag,
                                                      const double * grad, const char * I, const double GMin, const double Kii, const double tau,
                                                      int & Bj, double & GMax, double & GMax2, double & delta, SignNuType signNuType)
{
    double fpMax      = MaxVal<double>::get();
    double GMax2Local = -fpMax; // store min(-y[i]*grad[i]) or max(y[i]*grad[i]), y[i]*grad[i] = -GMin2
    double GMaxLocal  = -fpMax; // store min(-b^2/a) or max(b^2/a), b^2/a = -GMin

    double zero(0.0);
    double two(2.0);

    const char sign = getSign(signNuType);

    // generally, we find min(grad[i]).
    // for performance considerations, it is better to find max(y[i]*grad[i]) instead.
    __m512d valGMax2 = _mm512_set1_pd(GMax2Local);

    // generally, we find min(-b^2/a). For perf considerations, find max(b^2/a) instead.
    __m512d valGMax = _mm512_set1_pd(GMaxLocal);

    // constant used to select j so that ygrad = -y[i]*grad[i] < GMax.
    // for performance considerations, we select j so that -ygrad = y[i]*grad[i] >= GMax.
    __m512d valGMin = _mm512_set1_pd(GMin);

    // we minimize over index Bj. It is stored as vector.
    // We find 8 minimums in parallel to choose 1 later on reduction step.
    __m256i Bj_vec = _mm256_set1_epi32((int)-1);

    // some constants used during optimization
    // enum SVMVectorStatus low = 0x2
    __m128i vecSignLow =
        (signNuType == SignNuType::none) ? _mm_set1_epi8(low) : _mm_or_si128(_mm_set1_epi8(low), _mm_set1_epi8(sign)); // vector of masks
    __m512d two_vec = _mm512_set1_pd(two);                                                                             // vector of 2's
    __m512d Kii_vec = _mm512_set1_pd(Kii);                                                                             // vector of Kii's
    __m512d tau_vec = _mm512_set1_pd(tau);                                                                             // vector of tau's

    // mask_1: condition _I[j] is in I_low
    // mask_2: condition -ygrad = y[i]*grad[i] >= (GE) -GMax.
    // mask_3: condition a > tau

    size_t j_cur;

    unsigned short int cmp_all  = 65535; // all 16 bits are 1
    unsigned short int cmp_half = 65280; // first 8 bits are one

    __mmask16 mask_cmp = *((__mmask16 *)&cmp_all);

    for (j_cur = jStart; (j_cur + 8) <= jEnd; j_cur += 8)
    {
        // if (jEnd > jCur)  make mask?
        DAAL_ASSERT(j_cur <= services::internal::MaxVal<int>::get())
        __m256i Bj_vec_cur = _mm256_set1_epi32((int)(j_cur));

        if ((j_cur + 8) > jEnd) // last iteration
        {
            mask_cmp = *((__mmask16 *)&cmp_half);
        }
        __m128i vec_I = _mm_maskz_loadu_epi8(mask_cmp, &I[j_cur]); // basically, I load 16 chars except last iteration

        // vector grad[jcur:jcur+16] = _grad[jcur:jcur+16]
        __m512d valGrad = _mm512_load_pd(&grad[j_cur]);

        // if (!(I[j] & sign) || (_I[j] & low) != low) { continue; }
        __mmask16 mask1_tmp_ = _mm_mask_cmpeq_epi8_mask(mask_cmp, _mm_and_si128(vec_I, vecSignLow), vecSignLow);
        __mmask8 mask1_      = *((__mmask8 *)&mask1_tmp_); // take first half of mask

        // find max(y[j]*grad[j]) instead - here's where we find vector of max's.
        // result can be updated only if _I[j] is in I_low.
        valGMax2 = _mm512_mask_max_pd(valGMax2, mask1_, valGrad, valGMax2);

        // we select j so that -ygrad = y[i]*grad[i] >= (GE) -GMax.
        __mmask8 mask2_ = _mm512_cmp_pd_mask(valGrad, valGMin, _CMP_GT_OS); // if (ygrad < GMaxLocal) { continue; }
        mask2_          = mask1_ & mask2_;                                  // combine with previous mask
        __m512d b_vec   = _mm512_sub_pd(valGMin, valGrad);                  // b = Gmax - (-y*grad) = y*grad - (-Gmax)

        // float a = Kii + _kernelDiag[j] - two * KiBlock[j - jStart] = Kii + _kernelDiag[j] - KiBlock[j - jStart] - KiBlock[j - jStart];
        // a_tmp = two * KiBlock[j - jStart] - kernelDiag[j]
        __m512d KiBlock_vec    = _mm512_loadu_pd(KiBlock + j_cur - jStart);
        __m512d kernelDiag_vec = _mm512_load_pd(&kernelDiag[j_cur]);
        __m512d a_vec          = _mm512_fmsub_pd(two_vec, KiBlock_vec, kernelDiag_vec);

        // originally, if a < 0, a = tau
        // mask3_ : 1 if Kii > a_tmp
        // if mask3_ : 1, a = Kii - a_tmp, else a = tau.
        __mmask8 mask3_ = _mm512_cmp_pd_mask(Kii_vec, a_vec, _CMP_GT_OS);
        a_vec           = _mm512_mask_sub_pd(tau_vec, mask3_, Kii_vec, a_vec);

        __m512d objFunc_vec = _mm512_div_pd(b_vec, a_vec);       // b/a = delta.
        objFunc_vec         = _mm512_mul_pd(objFunc_vec, b_vec); // we have objFunc = b * b / a

        // generally, objFunc = -b^2/a.
        // if (objFunc <= GMin) -> if (-objFunc > -GMin)
        __mmask8 mask4_ = _mm512_cmp_pd_mask(objFunc_vec, valGMax, _CMP_GE_OS);
        mask4_          = mask2_ & mask4_;                                   // combine with previous masks
        valGMax         = _mm512_mask_mov_pd(valGMax, mask4_, objFunc_vec);  // -GMin = b^2/a
        Bj_vec          = _mm256_mask_mov_epi32(Bj_vec, mask4_, Bj_vec_cur); // result index
    }

    // reduction:
    const double * const GMaxVal      = reinterpret_cast<double * const>(&valGMax);
    const int * const BjValOverall    = reinterpret_cast<int * const>(&Bj_vec);
    const double * const GMax2Overall = reinterpret_cast<double * const>(&valGMax2);

    GMax  = -fpMax;
    GMax2 = -fpMax;

    for (size_t k = 0; k < 8; ++k)
    {
        if (GMaxVal[k] > GMax)
        {
            GMax = GMaxVal[k];
            Bj   = BjValOverall[k] + k; // from partial index to final
        }
        if (GMax2Overall[k] > GMax2)
        {
            GMax2 = GMax2Overall[k];
        }
    }

    if (Bj != -1)
    {
        const double gradBj = grad[Bj];
        const double b      = GMin - gradBj;
        double a            = Kii + kernelDiag[Bj] - two * KiBlock[Bj - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        delta = b / a;
        GMax  = b * delta;
    }
    else
    {
        GMax  = -fpMax;
        GMax2 = -fpMax;
    }
    delta = -delta;
    if (j_cur < jEnd) // we have tail
    {
        int Bj_local      = -1;
        double GMax_local = -fpMax, GMax2_local = -fpMax, delta_local;
        WSSjLocalBaseline(j_cur, jEnd, (KiBlock + j_cur - jStart), kernelDiag, grad, I, GMin, Kii, tau, Bj_local, GMax_local, GMax2_local,
                          delta_local, signNuType);
        if (GMax_local > GMax)
        {
            GMax  = GMax_local;
            Bj    = Bj_local;
            delta = delta_local;
        }
        if (GMax2_local > GMax2)
        {
            GMax2 = GMax2_local;
        }
    }
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
