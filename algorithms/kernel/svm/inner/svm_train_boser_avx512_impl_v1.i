/* file: svm_train_boser_avx512_impl_v1.i */
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
 * Contains optimizations for AVX512.
*/

#include "service_data_utils.h"

template <>
void SVMTrainTask<float, daal::algorithms::svm::interface1::Parameter, avx512>::WSSjLocal(const size_t jStart, const size_t jEnd,
                                                                                          const float * KiBlock, const float GMax, const float Kii,
                                                                                          const float tau, int & Bj, float & GMin, float & GMin2,
                                                                                          float & delta) const
{
    float fpMax      = MaxVal<float>::get();
    float minusGMin2 = -fpMax; // store min(-y[i]*grad[i]) or max(y[i]*grad[i]), y[i]*grad[i] = -GMin2
    float minusGMin  = -fpMax; // store min(-b^2/a) or max(b^2/a), b^2/a = -GMin
    float minusGMax  = -GMax;

    float zero(0.0);
    float two(2.0);

    // generally, we find min(-y[i]*grad[i]).
    // for performance considerations, it is better to find max(y[i]*grad[i]) instead.
    __m512 valMinusGMin2 = _mm512_set1_ps(minusGMin2);

    // generally, we find min(-b^2/a). For perf considerations, find max(b^2/a) instead.
    __m512 valMinusGMin = _mm512_set1_ps(minusGMin);

    // constant used to select j so that ygrad = -y[i]*grad[i] < GMax.
    // for performance considerations, we select j so that -ygrad = y[i]*grad[i] >= GMax.
    __m512 valMinusGMax = _mm512_set1_ps(minusGMax);

    // we minimize over index Bj. It is stored as vector.
    // We find 16 minimums in parallel to choose 1 later on reduction step.
    __m512i Bj_vec = _mm512_set1_epi32((int)-1);

    // some constants used during optimization
    // enum SVMVectorStatus low = 0x2
    __m128i vecAllLow = _mm_set1_epi8(low);  // vector of masks
    __m512 two_vec    = _mm512_set1_ps(two); // vector of 2's
    __m512 Kii_vec    = _mm512_set1_ps(Kii); // vector of Kii's
    __m512 tau_vec    = _mm512_set1_ps(tau); // vector of tau's

    // mask_1: condition _I[j] is in I_low
    // mask_2: condition -ygrad = y[i]*grad[i] >= (GE) -GMax.
    // mask_3: condition a > tau

    size_t j_cur;
    for (j_cur = jStart; (j_cur + 16) <= jEnd; j_cur += 16)
    {
        // if (jEnd > jCur)  make mask?
        DAAL_ASSERT(j_cur <= services::internal::MaxVal<int>::get())
        __m512i Bj_vec_cur = _mm512_set1_epi32((int)(j_cur));
        __m128i vec_I      = _mm_load_si128((__m128i *)(&_I[j_cur]));

        // vector multiplication y_grad[jcur:jcur+16] = _y[jcur:jcur+16] * _grad[jcur:jcur+16]
        __m512 valYGrad = _mm512_mul_ps(_mm512_loadu_ps(&_y[j_cur]), _mm512_loadu_ps(&_grad[j_cur]));

        // if ((_I[j] & low) != low) { continue; }
        __mmask16 mask1_ = _mm_cmpeq_epi8_mask(_mm_and_si128(vec_I, vecAllLow), vecAllLow);

        // find max(y[j]*grad[j]) instead - here's where we find vector of max's.
        // result can be updated only if _I[j] is in I_low.
        valMinusGMin2 = _mm512_mask_max_ps(valMinusGMin2, mask1_, valYGrad, valMinusGMin2);

        // we select j so that -ygrad = y[i]*grad[i] >= (GE) -GMax.
        __mmask16 mask2_ = _mm512_cmp_ps_mask(valYGrad, valMinusGMax, _CMP_GE_OS); // if (ygrad < minusGMax) { continue; }
        mask2_           = _mm512_kand(mask1_, mask2_);                            // combine with previous mask
        __m512 b_vec     = _mm512_sub_ps(valYGrad, valMinusGMax);                  // b = Gmax - (-y*grad) = y*grad - (-Gmax)

        // float a = Kii + _kernelDiag[j] - two * KiBlock[j - jStart] = Kii + _kernelDiag[j] - KiBlock[j - jStart] - KiBlock[j - jStart];
        // a_tmp = two * KiBlock[j - jStart] - kernelDiag[j]
        __m512 KiBlock_vec    = _mm512_loadu_ps(KiBlock + j_cur - jStart);
        __m512 kernelDiag_vec = _mm512_loadu_ps(&_kernelDiag[j_cur]);
        __m512 a_vec          = _mm512_fmsub_ps(two_vec, KiBlock_vec, kernelDiag_vec);

        // originally, if a < 0, a = tau
        // mask3_ : 1 if Kii > a_tmp
        // if mask3_ : 1, a = Kii - a_tmp, else a = tau.
        __mmask16 mask3_ = _mm512_cmp_ps_mask(Kii_vec, a_vec, _CMP_GT_OS);
        a_vec            = _mm512_mask_sub_ps(tau_vec, mask3_, Kii_vec, a_vec);

        __m512 objFunc_vec = _mm512_div_ps(b_vec, a_vec);       // b/a = delta.
        objFunc_vec        = _mm512_mul_ps(objFunc_vec, b_vec); // we have objFunc = b * b / a

        // generally, objFunc = -b^2/a.
        // if (objFunc <= GMin) -> if (-objFunc > -GMin)
        __mmask16 mask4_ = _mm512_cmp_ps_mask(objFunc_vec, valMinusGMin, _CMP_GE_OS);
        mask4_           = _mm512_kand(mask2_, mask4_);                           // combine with previous masks
        valMinusGMin     = _mm512_mask_mov_ps(valMinusGMin, mask4_, objFunc_vec); // -GMin = b^2/a
        Bj_vec           = _mm512_mask_mov_epi32(Bj_vec, mask4_, Bj_vec_cur);     // result index
    }

    // reduction:
    TNArray<float, 16, avx512> minus_GMin_val(16);
    TNArray<int, 16, avx512> Bj_val_overall(16);
    TNArray<float, 16, avx512> minusGMin2_overall(16);

    _mm512_storeu_ps(minus_GMin_val.get(), valMinusGMin);
    _mm512_storeu_si512(Bj_val_overall.get(), Bj_vec);
    _mm512_storeu_ps(minusGMin2_overall.get(), valMinusGMin2);

    GMin  = -fpMax;
    GMin2 = -fpMax;

    for (size_t k = 0; k < 16; ++k)
    {
        if (minus_GMin_val[k] > GMin)
        {
            GMin = minus_GMin_val[k];
            Bj   = Bj_val_overall[k] + k; // from partial index to final
        }
        if (minusGMin2_overall[k] > GMin2)
        {
            GMin2 = minusGMin2_overall[k];
        }
    }

    if (Bj != -1)
    {
        float ygrad = -_y[Bj] * _grad[Bj];
        float b     = GMax - ygrad;
        float a     = Kii + _kernelDiag[Bj] - two * KiBlock[Bj - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        delta = b / a;
        GMin  = -b * delta;

        GMin2 = -GMin2;
    }
    else
    {
        GMin  = fpMax;
        GMin2 = fpMax;
    }

    if (j_cur < jEnd) // we have tail
    {
        int Bj_local = -1;
        float GMin_local, GMin2_local, delta_local;
        WSSjLocalBaseline(j_cur, jEnd, (KiBlock + j_cur - jStart), GMax, Kii, tau, Bj_local, GMin_local, GMin2_local, delta_local);
        if (GMin_local <= GMin)
        {
            GMin  = GMin_local;
            Bj    = Bj_local;
            delta = delta_local;
        }
        if (GMin2_local <= GMin2)
        {
            GMin2 = GMin2_local;
        }
    }
}

template <>
void SVMTrainTask<double, daal::algorithms::svm::interface1::Parameter, avx512>::WSSjLocal(const size_t jStart, const size_t jEnd,
                                                                                           const double * KiBlock, const double GMax,
                                                                                           const double Kii, const double tau, int & Bj,
                                                                                           double & GMin, double & GMin2, double & delta) const
{
    double fpMax      = MaxVal<double>::get();
    double minusGMin2 = -fpMax; // store min(-y[i]*grad[i]) or max(y[i]*grad[i]), y[i]*grad[i] = -GMin2
    double minusGMin  = -fpMax; // store min(-b^2/a) or max(b^2/a), b^2/a = -GMin
    double minusGMax  = -GMax;

    double zero(0.0);
    double two(2.0);

    // generally, we find min(-y[i]*grad[i]).
    // for performance considerations, it is better to find max(y[i]*grad[i]) instead.
    __m512d valMinusGMin2 = _mm512_set1_pd(minusGMin2);

    // generally, we find min(-b^2/a). For perf considerations, find max(b^2/a) instead.
    __m512d valMinusGMin = _mm512_set1_pd(minusGMin);

    // constant used to select j so that ygrad = -y[i]*grad[i] < GMax.
    // for performance considerations, we select j so that -ygrad = y[i]*grad[i] >= GMax.
    __m512d valMinusGMax = _mm512_set1_pd(minusGMax);

    // we minimize over index Bj. It is stored as vector.
    // We find 8 minimums in parallel to choose 1 later on reduction step.
    __m256i Bj_vec = _mm256_set1_epi32((int)-1);

    // some constants used during optimization
    // enum SVMVectorStatus low = 0x2
    __m128i vecAllLow = _mm_set1_epi8(low);  // vector of masks
    __m512d two_vec   = _mm512_set1_pd(two); // vector of 2's
    __m512d Kii_vec   = _mm512_set1_pd(Kii); // vector of Kii's
    __m512d tau_vec   = _mm512_set1_pd(tau); // vector of tau's

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
        __m128i vec_I = _mm_maskz_loadu_epi8(mask_cmp, &_I[j_cur]); // basically, I load 16 chars except last iteration

        // vector multiplication y_grad[jcur:jcur+8] = _y[jcur:jcur+8] * _grad[jcur:jcur+8]
        __m512d valYGrad = _mm512_mul_pd(_mm512_loadu_pd(&_y[j_cur]), _mm512_loadu_pd(&_grad[j_cur]));

        // if ((_I[j] & low) != low) { continue; }
        __mmask16 mask1_tmp_ = _mm_mask_cmpeq_epi8_mask(mask_cmp, _mm_and_si128(vec_I, vecAllLow), vecAllLow);
        __mmask8 mask1_      = *((__mmask8 *)&mask1_tmp_); // take first half of mask

        // find max(y[j]*grad[j]) instead - here's where we find vector of max's.
        // result can be updated only if _I[j] is in I_low.
        valMinusGMin2 = _mm512_mask_max_pd(valMinusGMin2, mask1_, valYGrad, valMinusGMin2);

        // we select j so that -ygrad = y[i]*grad[i] >= (GE) -GMax.
        __mmask8 mask2_ = _mm512_cmp_pd_mask(valYGrad, valMinusGMax, _CMP_GE_OS); // if (ygrad < minusGMax) { continue; }
        mask2_          = mask1_ & mask2_;                                        // combine with previous mask
        __m512d b_vec   = _mm512_sub_pd(valYGrad, valMinusGMax);                  // b = Gmax - (-y*grad) = y*grad - (-Gmax)

        // float a = Kii + _kernelDiag[j] - two * KiBlock[j - jStart] = Kii + _kernelDiag[j] - KiBlock[j - jStart] - KiBlock[j - jStart];
        // a_tmp = two * KiBlock[j - jStart] - kernelDiag[j]
        __m512d KiBlock_vec    = _mm512_loadu_pd(KiBlock + j_cur - jStart);
        __m512d kernelDiag_vec = _mm512_loadu_pd(&_kernelDiag[j_cur]);
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
        __mmask8 mask4_ = _mm512_cmp_pd_mask(objFunc_vec, valMinusGMin, _CMP_GE_OS);
        mask4_          = mask2_ & mask4_;                                       // combine with previous masks
        valMinusGMin    = _mm512_mask_mov_pd(valMinusGMin, mask4_, objFunc_vec); // -GMin = b^2/a
        Bj_vec          = _mm256_mask_mov_epi32(Bj_vec, mask4_, Bj_vec_cur);     // result index
    }

    // reduction:
    TNArray<double, 8, avx512> minus_GMin_val(8);
    TNArray<int, 8, avx512> Bj_val_overall(8);
    TNArray<double, 8, avx512> minusGMin2_overall(8);

    unsigned char ones_all = 255; // all 8 bits are 1
    __mmask8 maskOnes      = *((__mmask8 *)&ones_all);

    _mm512_storeu_pd(minus_GMin_val.get(), valMinusGMin);
    _mm256_mask_storeu_epi32(Bj_val_overall.get(), maskOnes, Bj_vec);
    _mm512_storeu_pd(minusGMin2_overall.get(), valMinusGMin2);

    GMin  = -fpMax;
    GMin2 = -fpMax;

    for (size_t k = 0; k < 8; ++k)
    {
        if (minus_GMin_val[k] > GMin)
        {
            GMin = minus_GMin_val[k];
            Bj   = Bj_val_overall[k] + k; // from partial index to final
        }
        if (minusGMin2_overall[k] > GMin2)
        {
            GMin2 = minusGMin2_overall[k];
        }
    }

    if (Bj != -1)
    {
        double ygrad = -_y[Bj] * _grad[Bj];
        double b     = GMax - ygrad;
        double a     = Kii + _kernelDiag[Bj] - two * KiBlock[Bj - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        delta = b / a;
        GMin  = -b * delta;

        GMin2 = -GMin2;
    }
    else
    {
        GMin  = fpMax;
        GMin2 = fpMax;
    }

    if (j_cur < jEnd) // we have tail
    {
        int Bj_local = -1;
        double GMin_local, GMin2_local, delta_local;
        WSSjLocalBaseline(j_cur, jEnd, (KiBlock + j_cur - jStart), GMax, Kii, tau, Bj_local, GMin_local, GMin2_local, delta_local);
        if (GMin_local <= GMin)
        {
            GMin  = GMin_local;
            Bj    = Bj_local;
            delta = delta_local;
        }
        if (GMin2_local <= GMin2)
        {
            GMin2 = GMin2_local;
        }
    }
}
