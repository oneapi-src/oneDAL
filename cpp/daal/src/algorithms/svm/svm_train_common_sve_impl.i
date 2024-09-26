/*******************************************************************************
* Copyright contributors to the oneDAL project
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
 * Contains optimizations for SVE.
*/

#include <arm_sve.h>
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
inline void HelperTrainSVM<float, sve>::WSSjLocal(const size_t jStart, const size_t jEnd, const float * KiBlock, const float * kernelDiag,
                                                  const float * grad, const char * I, const float GMin, const float Kii, const float tau, int & Bj,
                                                  float & GMax, float & GMax2, float & delta, SignNuType signNuType)
{
    const int w      = (int)svcntw(); //vector length
    float fpMax      = MaxVal<float>::get();
    float GMax2Local = -fpMax; // store min(grad[i]) or max(y[i]*grad[i]), y[i]*grad[i] = -GMin2
    float GMaxLocal  = -fpMax; // store min(-b^2/a) or max(b^2/a), b^2/a = -GMin
    float GMinLocal  = GMin;

    float zero(0.0);
    float two(2.0);

    const char sign = getSign(signNuType);

    svbool_t pgf = svptrue_b32(); //predicate for float

    svfloat32_t valGMax2 = svdup_f32(GMax2Local);
    svfloat32_t valGMax  = svdup_f32(GMaxLocal);
    svfloat32_t valGMin  = svdup_f32(GMinLocal);
    svint32_t Bj_vec     = svdup_s32(-1);

    // some constants used during optimization
    // enum SVMVectorStatus low = 0x2
    svint32_t vecSignLow;
    if (signNuType == SignNuType::none)
    {
        vecSignLow = svdup_n_s32(low);
    }
    else
    {
        DAAL_ASSERT((sign & (sign - 1)) == 0) // used to make sure sign is always having 1 bit set
        svint32_t t1 = svdup_n_s32(low);
        svint32_t t2 = svdup_n_s32(sign);
        vecSignLow   = svorr_s32_z(pgf, t1, t2);
    }

    svfloat32_t two_vec = svdup_f32(two);
    svfloat32_t Kii_vec = svdup_f32(Kii);
    svfloat32_t tau_vec = svdup_f32(tau);

    for (size_t j_cur = jStart; j_cur < jEnd; j_cur += w)
    {
        svint32_t Bj_vec_cur = svindex_s32(j_cur, 1);      // Bj value starts with j_cur
        svbool_t pg2         = svwhilelt_b32(j_cur, jEnd); // adapts to vector length

        svint32_t vec_I = svld1sb_s32(pg2, reinterpret_cast<const int8_t *>(&I[j_cur])); // load chars

        // Combine 2 if conditions
        // cond1: !(I[j]&sign) {continue}
        // cond2: (I[j]&low)!=low {continue}
        // combined: (I[j] & (sign | low)) == (sign | low)
        // assertion @L63 is a prerequisite for the combined condition to satisfy
        svint32_t result_of_and32 = svand_s32_m(pg2, vec_I, vecSignLow);
        pg2                       = svcmpeq_s32(pg2, result_of_and32, vecSignLow); // if pg2 bit is 0 then continue;

        svfloat32_t valGrad = svld1_f32(pg2, &grad[j_cur]); // load grads
        // if (gradj > GMax2) { GMax2 = gradj; }
        valGMax2 = svmax_f32_m(pg2, valGMax2, valGrad);
        // cond3: if (gradj < GMin) { continue; }
        svbool_t cond3 = svcmpge_f32(pg2, valGrad, valGMin);
        pg2            = svand_b_z(pg2, pg2, cond3); // combine all 3 conditions

        svfloat32_t b_vec = svsub_f32_x(pg2, valGMin, valGrad); // b = Gmin - grad

        svfloat32_t KiBlock_vec    = svld1_f32(pg2, KiBlock + j_cur - jStart);                // load kiBlocs
        svfloat32_t kernelDiag_vec = svld1_f32(pg2, &kernelDiag[j_cur]);                      // load kernelDiags
        svfloat32_t a_vec          = svnmls_f32_x(pg2, kernelDiag_vec, two_vec, KiBlock_vec); // a_tmp = two * KiBlock[j - jStart] - kernelDiag[j]

        // originally, if a < 0, a = tau
        // mask3_ : 1 if Kii > a_tmp
        // if mask3_ : 1, a = Kii - a_tmp, else a = tau.
        svbool_t mask3_ = svcmpgt_f32(pg2, Kii_vec, a_vec);
        a_vec           = svsel_f32(mask3_, svsub_f32_x(mask3_, Kii_vec, a_vec), tau_vec);

        svfloat32_t dt_vec      = svdiv_f32_x(pg2, b_vec, a_vec);  // b/a = delta.
        svfloat32_t objFunc_vec = svmul_f32_x(pg2, dt_vec, b_vec); // objFunc = b * delta

        svbool_t mask4_ = svcmpgt_f32(pg2, objFunc_vec, valGMax);  // if (objFunc > GMax)
        valGMax         = svsel_f32(mask4_, objFunc_vec, valGMax); // if mask is 1, valGMax = objFunc_vec, else valGMax original value
        Bj_vec          = svsel_s32(mask4_, Bj_vec_cur, Bj_vec);   // if mask is 1, Bj_vec = Bj_vec_cur, else Bj_vec original value
    }

    // reductions
    GMax              = svmaxv_f32(pgf, valGMax);
    GMax2             = svmaxv_f32(pgf, valGMax2);
    svbool_t tmp_mask = svcmpeq(pgf, svdup_f32(GMax), valGMax);
    Bj                = svmaxv_s32(tmp_mask, Bj_vec);

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
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
