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

    inline bool isUpper(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
    }

    inline bool isLower(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
    }

    typedef struct {
        int index;
        algorithmFPType value;
    } KeyValue;

    void reduceArgMax(const __local algorithmFPType * values, __local KeyValue * localCache, __local KeyValue * result) {
        const int localGroupId = get_sub_group_local_id();
        const int groupId      = get_sub_group_id();
        const int localId      = get_local_id(0);
        const int groupCount   = get_num_sub_groups();

        algorithmFPType x = values[localId];
        int indX          = localId;

        algorithmFPType resMax;
        int resIndex;
        resMax   = sub_group_reduce_max(x);
        resIndex = sub_group_reduce_min(resMax == x ? indX : INT_MAX);

        if (localGroupId == 0)
        {
            localCache[groupId].value = resMax;
            localCache[groupId].index = resIndex;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (groupId == 0 && localGroupId < groupCount)
        {
            x        = localCache[localGroupId].value;
            indX     = localCache[localGroupId].index;
            resMax   = sub_group_reduce_max(x);
            resIndex = sub_group_reduce_min(resMax == x ? indX : INT_MAX);

            if (localGroupId == 0)
            {
                result->value = resMax;
                result->index = resIndex;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __kernel void smoKernel(const __global algorithmFPType * const y, const __global algorithmFPType * const kernelWsRows,
                            const __global int * wsIndices, const uint nVectors, const __global algorithmFPType * grad, const algorithmFPType C,
                            const algorithmFPType eps, const algorithmFPType tau, const uint maxInnerIteration, __global algorithmFPType * alpha,
                            __global algorithmFPType * deltaalpha, __global algorithmFPType * resinfo) {
        const uint i = get_local_id(0);
        __local algorithmFPType kd[WS_SIZE];

        const uint wsIndex = wsIndices[i];

        const algorithmFPType MIN_FLT = -FLT_MAX;

        const algorithmFPType two = (algorithmFPType)2.0;

        algorithmFPType gradi           = grad[wsIndex];
        algorithmFPType alphai          = alpha[wsIndex];
        const algorithmFPType oldalphai = alphai;
        const algorithmFPType yi        = y[wsIndex];

        __local algorithmFPType objFunc[WS_SIZE];

        __local algorithmFPType deltaBi;
        __local algorithmFPType deltaBj;

        __local KeyValue localCache[SIMD_WIDTH];
        __local KeyValue maxValInd;

        uint Bi = 0;
        uint Bj = 0;

        algorithmFPType ma;

        kd[i] = kernelWsRows[i * nVectors + wsIndex];
        barrier(CLK_LOCAL_MEM_FENCE);

        __local algorithmFPType localDiff;
        __local algorithmFPType localEps;

        uint iter = 0;
        for (; iter < maxInnerIteration; iter++)
        {
            /* m(alpha) = min(grad[i]): i belongs to I_UP (alpha) */
            objFunc[i] = isUpper(alphai, yi, C) ? -gradi : MIN_FLT;

            /* Find i index of the working set (Bi) */
            reduceArgMax(objFunc, localCache, &maxValInd);
            Bi = maxValInd.index;
            ma = -maxValInd.value;

            /* maxgrad(alpha) = max(grad[i]): i belongs to I_low (alpha) */
            objFunc[i] = isLower(alphai, yi, C) ? gradi : MIN_FLT;

            /* Find max gradinet */
            reduceArgMax(objFunc, localCache, &maxValInd);

            if (i == 0)
            {
                const algorithmFPType maxGrad = maxValInd.value;

                /* for condition check: m(alpha) >= maxgrad */
                localDiff = maxGrad - ma;
                if (iter == 0)
                {
                    localEps   = max(eps, localDiff * (algorithmFPType)1e-1);
                    resinfo[1] = localDiff;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if (localDiff < localEps)
            {
                break;
            }

            const algorithmFPType Kii   = kd[i];
            const algorithmFPType KBiBi = kd[Bi];
            const algorithmFPType KiBi  = kernelWsRows[Bi * nVectors + wsIndex];

            if (isLower(alphai, yi, C) && ma < gradi)
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

            /* Find j index of the working set (Bj) */
            reduceArgMax(objFunc, localCache, &maxValInd);
            Bj = maxValInd.index;

            const algorithmFPType KiBj = kernelWsRows[Bj * nVectors + wsIndex];

            /* Update alpha */
            if (i == Bi)
            {
                deltaBi = yi > 0 ? C - alphai : alphai;
            }
            if (i == Bj)
            {
                deltaBj                 = yi > 0 ? alphai : C - alphai;
                const algorithmFPType b = ma - gradi;
                const algorithmFPType a = max(Kii + KBiBi - two * KiBi, tau);

                const algorithmFPType dt = -b / a;
                deltaBj                  = min(deltaBj, dt);
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

            /* Update gradient */
            gradi = gradi + delta * (KiBi - KiBj);
        }
        alpha[wsIndex] = alphai;
        deltaalpha[i]  = (alphai - oldalphai) * yi;
        if (i == 0)
        {
            resinfo[0] = iter;
        }
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
