/* file: svm_train_block_smo_oneapi.cl */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of SMO algorithm for wokset block.
//--
*/

#ifndef __SVM_TRAIN_BLOCK_SMO_ONEAPI_CL__
#define __SVM_TRAIN_BLOCK_SMO_ONEAPI_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char *(name) = #src;

DECLARE_SOURCE_DAAL(
    clKernelBlockSMO,

    inline bool IUpper(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
    }

    inline bool ILower(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
    }

    void reduceMax(const __local algorithmFPType * values, __local int * indices) {
        const int group_size = get_local_size(0);
        const int local_id   = get_local_id(0);

        indices[local_id] = local_id;

        for (int stride = group_size / 2; stride > 0; stride >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            if (local_id < stride)
            {
                const algorithmFPType v  = values[indices[local_id]];
                const algorithmFPType vk = values[indices[local_id + stride]];
                if (vk > v)
                {
                    indices[local_id] = indices[local_id + stride];
                }
            }
        }
    }

    void reduceMin(const __local algorithmFPType * values, __local int * indices) {
        const int group_size = get_local_size(0);
        const int local_id   = get_local_id(0);

        indices[local_id] = local_id;

        for (int stride = group_size / 2; stride > 0; stride >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            if (local_id < stride)
            {
                const algorithmFPType v  = values[indices[local_id]];
                const algorithmFPType vk = values[indices[local_id + stride]];
                if (vk < v)
                {
                    indices[local_id] = indices[local_id + stride];
                }
            }
        }
    }

    __kernel void smoKernel(const __global algorithmFPType * const y, const __global algorithmFPType * const kernelWsRows,
                            const __global int * wsIndices, const uint ldx, const __global algorithmFPType * grad, const algorithmFPType C,
                            const algorithmFPType eps, const algorithmFPType tau, const int maxInnerIteration, __global algorithmFPType * alpha,
                            __global algorithmFPType * deltaalpha, __global algorithmFPType * resinfo) {
        const uint i = get_local_id(0);

        __local algorithmFPType kd[WS_SIZE];

        const int wsIndex = wsIndices[i];

        const algorithmFPType MIN_FLT = -1e20;
        const algorithmFPType MAX_FLT = 1e20;

        const algorithmFPType two = 2.0;

        algorithmFPType gradi           = grad[wsIndex];
        algorithmFPType alphai          = alpha[wsIndex];
        const algorithmFPType oldalphai = alphai;
        const algorithmFPType yi        = y[wsIndex];

        __local algorithmFPType objFunc[WS_SIZE];
        __local int indices[WS_SIZE];

        __local algorithmFPType deltaBi;
        __local algorithmFPType deltaBj;

        __local int Bi;
        __local int Bj;

        algorithmFPType ma;

        kd[i] = kernelWsRows[i * ldx + wsIndex];
        barrier(CLK_LOCAL_MEM_FENCE);

        __local algorithmFPType localDiff;
        __local algorithmFPType localEps;

        int iter = 0;
        for (; iter < maxInnerIteration; iter++)
        {
            /* m(alpha) = min(grad[i]): i belongs to I_UP (alpha) */
            objFunc[i] = IUpper(alphai, yi, C) ? gradi : MAX_FLT;

            /* Find i index of the working set (Bi) */
            reduceMin(objFunc, indices);
            if (i == 0)
            {
                Bi = indices[0];
            }
            ma = objFunc[Bi];

            barrier(CLK_LOCAL_MEM_FENCE);

            /* maxgrad(alpha) = max(grad[i]): i belongs to I_low (alpha) */
            objFunc[i] = ILower(alphai, yi, C) ? gradi : MIN_FLT;

            /* Find max gradinet of the working set (Bi) */
            reduceMax(objFunc, indices);

            barrier(CLK_LOCAL_MEM_FENCE);
            if (i == 0)
            {
                const algorithmFPType maxGrad = objFunc[indices[0]];

                /* for condition check: m(alpha) >= maxgrad */
                localDiff = maxGrad - ma;
                if (iter == 0)
                {
                    localEps = max(eps, localDiff * (algorithmFPType)5e-1);
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if (localDiff < localEps)
            {
                break;
            }

            const algorithmFPType Kii   = kd[i];
            const algorithmFPType KBiBi = kd[Bi];
            const algorithmFPType KiBi  = kernelWsRows[Bi * ldx + wsIndex];

            if (ILower(alphai, yi, C) && ma < gradi)
            {
                /* M(alpha) = max((b^2/a) : i belongs to I_low(alpha) and ma < grad(alpha) */
                const algorithmFPType b  = ma - gradi;
                const algorithmFPType a  = max(Kii + KBiBi - two * KiBi, tau);
                const algorithmFPType dt = b / a;

                objFunc[i] = b * dt;
            }
            else
            {
                objFunc[i] = MIN_FLT;
            }

            // printf("> objFunc %.2f ygrad %.2f ma %.2f alphai %.2f C %.2f -b * dt %.2f wsIndex %d i %d\n", objFunc[i], gradi, ma, alphai, C, -b * dt, wsIndex, i);

            /* Find j index of the working set (Bj) */
            reduceMax(objFunc, indices);

            if (i == 0)
            {
                Bj = indices[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType KiBj = kernelWsRows[Bj * ldx + wsIndex];

            // Update alpha

            if (i == Bi)
            {
                deltaBi = yi > 0 ? C - alphai : alphai;
                // printf("> deltaBi %.3f\n", deltaBi);
                // printf("> Bi %d \n", Bi);
                // printf("> Ma %.f ygrad %.2f ma %.2f alphai %.2f yi %.2f C\n", Ma, gradi, ma, alphai, yi, C);
            }
            if (i == Bj)
            {
                deltaBj                 = yi > 0 ? alphai : C - alphai;
                const algorithmFPType b = ma - gradi;
                const algorithmFPType a = max(Kii + KBiBi - two * KiBi, tau);

                const algorithmFPType dt = -b / a;
                deltaBj                  = min(deltaBj, dt);
                // printf("> deltaBj %.3f dt %.3f b %.3f a %.3f\n", deltaBj, dt, b, a);
                // printf(" Bj %d \n", Bj);
                // printf(" curEps %f\n",curEps);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType delta = min(deltaBi, deltaBj);
            if (i == Bi)
            {
                alphai = alphai + yi * delta;
            }
            if (i == Bj)
            {
                alphai = alphai - yi * delta;
            }

            // Update gradient
            gradi = gradi + delta * (KiBi - KiBj);
            // printf("> alphai %.3f gradi %.4f delta %.4f Bi %d Bj %d wsIndex %d i %d\n", alphai, gradi, delta, Bi, Bj, wsIndex, i);
            // printf("> Kii %.3f i %d\n", Kii, i);
        }
        alpha[wsIndex] = alphai;
        deltaalpha[i]  = (alphai - oldalphai) * yi;
        if (i == 0)
        {
            resinfo[0] = iter;
            resinfo[1] = localDiff;
        }
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
