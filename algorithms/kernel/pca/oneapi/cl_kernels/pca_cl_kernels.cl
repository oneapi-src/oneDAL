/* file: pca_cl_kernels.cl */
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
//  Implementation of PCA OpenCL kernels.
//--
*/

#ifndef __PCA_CL_KERNELS_CL__
#define __PCA_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    pca_cl_kernels,

    __kernel void calculateVariances(__global algorithmFPType * covariance, __global algorithmFPType * variances) {
        const int tid       = get_global_id(0);
        const int nFeatures = get_global_size(0);

        variances[tid] = covariance[tid * nFeatures + tid];
    }

    __kernel void range(__global int * x) {
        const int i = get_global_id(0);
        x[i]        = i;
    }

    inline bool inUpper(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        // (0 < a && a < C) || (y == 1  && a == 0) || (y == -1 && a == C);
        return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
    }

    inline bool inLower(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        // (0 < a && a < C) || (y == -1 && a == 0) || (y == 1 && a == C);
        return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
    }

    __kernel void checkUpper(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global int * indicator) {
        const int i  = get_global_id(0);
        indicator[i] = inUpper(alpha[i], y[i], C);
    }

    __kernel void checkLower(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global int * indicator) {
        const int i  = get_global_id(0);
        indicator[i] = inLower(alpha[i], y[i], C);
    }

    __kernel void copyBlockIndices(const __global algorithmFPType * const x, const __global int * const ind, const uint ldx,
                                   __global algorithmFPType * newX) {
        const uint index = get_global_id(1);
        const uint jCol  = get_global_id(0);

        const int iRow = ind[index];

        const __global algorithmFPType * const xi = &x[iRow * ldx];
        __global algorithmFPType * newXi          = &newX[index * ldx];

        newXi[jCol] = xi[jCol];
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
                if (vk >= v)
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
                if (vk <= v)
                {
                    indices[local_id] = indices[local_id + stride];
                }
            }
        }
    }

    // algorithmFPType WSSi(const algorithmFPType gradi, const algorithmFPType alphai, const algorithmFPType yi, const algorithmFPType C, int * Bi) {
    //     const uint i = get_local_id(0);

    //     // TODO
    //     const algorithmFPType MIN_FLT = -1e20;

    //     *Bi = -1;
    //     __local algorithmFPType objFunc[WS_SIZE];
    //     __local int indices[WS_SIZE];

    //     objFunc[i] = inUpper(alphai, yi, C) ? -yi * gradi : MIN_FLT;

    //     /* Find i index of the working set (Bi) */
    //     // reduceMax(objFunc, indices);
    //     barrier(CLK_LOCAL_MEM_FENCE);
    //     *Bi                         = indices[0];
    //     const algorithmFPType GMax = objFunc[*Bi];

    //     return GMax;
    // }

    // algorithmFPType WSSj(const algorithmFPType gradi, const algorithmFPType alphai, const algorithmFPType yi, const algorithmFPType Kii,
    //                      const algorithmFPType KBiBi, const algorithmFPType KiBi, const algorithmFPType tau, const algorithmFPType GMax, int & Bj) {
    //     const uint i = get_local_id(0);

    //     Bj = -1;

    //     __local algorithmFPType objFunc[WS_SIZE];
    //     __local int indices[WS_SIZE];

    //     // TODO
    //     const algorithmFPType MAX_FLT = 1e20;

    //     const algorithmFPType zero = 0.0;
    //     const algorithmFPType two  = 2.0;

    //     const algorithmFPType ygrad = -yi * gradi;

    //     const algorithmFPType b = GMax - ygrad;
    //     const algorithmFPType a = max(Kii + KBiBi - two * KiBi, tau);

    //     const algorithmFPType dt = b / a;

    //     objFunc[i] = inLower(alphai, yi, C) && ygrad < GMax ? -b * dt : MAX_FLT;

    //     reduceMin(objFunc, indices);
    //     barrier(CLK_LOCAL_MEM_FENCE);
    //     Bj                         = indices[0];
    //     const algorithmFPType GMin = objFunc[Bj];

    //     return GMin;
    // }

    __kernel void smoKernel(const __global algorithmFPType * const y, const __global algorithmFPType * const kernelWsRows,
                            const __global int * wsIndices, const uint ldx, const __global algorithmFPType * grad, const algorithmFPType C,
                            const algorithmFPType eps, const algorithmFPType tau, const int maxInnerIteration, __global algorithmFPType * alpha,
                            __global algorithmFPType * deltaalpha, __global algorithmFPType * resinfo) {
        const uint i = get_local_id(0);

        __local algorithmFPType kd[WS_SIZE];

        const int wsIndex = wsIndices[i];

        const algorithmFPType MIN_FLT = -1e20;
        const algorithmFPType MAX_FLT = 1e20;

        const algorithmFPType zero = 0.0;
        const algorithmFPType two  = 2.0;

        algorithmFPType gradi     = grad[wsIndex];
        algorithmFPType alphai    = alpha[wsIndex];
        algorithmFPType oldalphai = alphai;
        const algorithmFPType yi  = y[wsIndex];

        __local algorithmFPType objFunc[WS_SIZE];
        __local int indices[WS_SIZE];

        __local algorithmFPType deltaBi;
        __local algorithmFPType deltaBj;

        kd[i] = kernelWsRows[i * ldx + wsIndex];

        int iter = 0;
        for (; iter < 2/*maxInnerIteration*/; iter++)
        {
            /* m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha) */
            objFunc[i] = inUpper(alphai, yi, C) ? -yi * gradi : MIN_FLT;

            /* Find i index of the working set (Bi) */
            reduceMax(objFunc, indices);
            barrier(CLK_LOCAL_MEM_FENCE);
            int Bi                         = indices[0];
            const algorithmFPType ma = objFunc[Bi];

            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType Kii   = kd[i];
            const algorithmFPType KBiBi = kd[Bi];
            const algorithmFPType KiBi  = kernelWsRows[Bi * ldx + wsIndex];

            /* Find j index of the working set (Bj) */
            const algorithmFPType ygrad = -yi * gradi;

            const algorithmFPType b = ma - ygrad;
            const algorithmFPType a = max(Kii + KBiBi - two * KiBi, tau);

            const algorithmFPType dt = b / a;

            objFunc[i] = inLower(alphai, yi, C) && ygrad <= ma ? -b * dt : MAX_FLT;

            // printf("> objFunc %.2f ygrad %.2f ma %.2f alphai %.2f yi %.2f C %.2f -b * dt %.2f wsIndex %d i %d\n", objFunc[i], ygrad, ma, alphai, yi, C, -b * dt, wsIndex, i);

            reduceMin(objFunc, indices);
            barrier(CLK_LOCAL_MEM_FENCE);
            int Bj                         = indices[0];
            const algorithmFPType Ma = objFunc[Bj];

            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType KiBj = kernelWsRows[Bj * ldx + wsIndex];

            /* ma - Ma is used to check stopping condition */
            const algorithmFPType curEps = ma - Ma;

            printf("> curEps %.3f Bi %d ma %.4f Bj %d Ma %.4f wsIndex %d i %d KiBj %.2f\n", curEps, Bi, ma, Bj, Ma, wsIndex, i, KiBj);


            if (curEps < 10.0 * eps)
            {
                resinfo[1] = curEps;
                break;
            }
            // Update alpha

            if (i == Bi)
            {
                deltaBi = yi > 0 ? C - alphai : alphai;
                printf("> deltaBi %.3f\n", deltaBi);

            }
            if (i == Bj)
            {
                deltaBj                     = yi > 0 ? alphai : C - alphai;
                const algorithmFPType ygrad = -yi * gradi;
                const algorithmFPType b     = ma - ygrad;
                const algorithmFPType a     = max(kd[i] + kd[Bi] - two * KiBj, tau);

                const algorithmFPType dt = b / a;
                deltaBj                  = min(deltaBj, dt);
                printf("> deltaBj %.3f dt %.3f\n", deltaBj, dt);
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
            printf("> alphai %.3f gradi %.4f delta %.4f wsIndex %d i %d\n", alphai, gradi, delta, wsIndex, i);


        }
        alpha[wsIndex] = alphai;
        deltaalpha[i]  = (oldalphai - alphai) * yi;
        resinfo[0]     = iter;
    }

);

#endif
