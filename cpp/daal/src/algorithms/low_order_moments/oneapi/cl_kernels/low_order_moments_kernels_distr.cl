/* file: low_order_moments_kernels_distr.cl */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
//  Implementation of low order moments kernels.
//--
*/

#define CONCAT(n, suff) n##suff
#define FULLNAME(n, p)  CONCAT(n, p)

#define mergeDistrBlocks FULLNAME(mergeDistrBlocks, FNAMESUFF)
#define finalize         FULLNAME(finalize, FNAMESUFF)

/* merge distributed blocks kernel */
__kernel void mergeDistrBlocks(uint nFeatures, uint nBlocks, uint stride
#if (defined _RMIN_)
                               ,
                               __global algorithmFPType * gMin
#endif
#if (defined _RMAX_)
                               ,
                               __global algorithmFPType * gMax
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                               ,
                               __global algorithmFPType * gSum
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
                               ,
                               __global algorithmFPType * gSum2
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                               ,
                               __global algorithmFPType * gSum2Cent
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                               ,
                               const __global algorithmFPType * bNVec
#endif
#if (defined _RMIN_)
                               ,
                               const __global algorithmFPType * bMin
#endif
#if (defined _RMAX_)
                               ,
                               const __global algorithmFPType * bMax
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                               ,
                               const __global algorithmFPType * bSum
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
                               ,
                               const __global algorithmFPType * bSum2
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                               ,
                               const __global algorithmFPType * bSum2Cent
#endif
)
{
    const uint itemId = get_global_id(0);

#if (defined _RMIN_)
    algorithmFPType mrgMin = bMin[itemId];
#endif
#if (defined _RMAX_)
    algorithmFPType mrgMax = bMax[itemId];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType mrgSum = (algorithmFPType)0;
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
    algorithmFPType mrgSum2 = (algorithmFPType)0;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType mrgVectors  = (algorithmFPType)0;
    algorithmFPType mrgSum2Cent = (algorithmFPType)0;
    algorithmFPType mrgMean     = (algorithmFPType)0;
#endif

    for (uint i = 0; i < nBlocks; i++)
    {
        uint offset = i * stride;

#if (defined _RMIN_)
        algorithmFPType min = bMin[offset];
#endif
#if (defined _RMAX_)
        algorithmFPType max = bMax[offset];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType sum = bSum[offset];
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        algorithmFPType sum2 = bSum2[offset];
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType nVec     = bNVec[i];
        algorithmFPType sum2Cent = bSum2Cent[offset];
        algorithmFPType mean     = sum / nVec;

        algorithmFPType sumN1N2    = mrgVectors + nVec;
        algorithmFPType mulN1N2    = mrgVectors * nVec;
        algorithmFPType deltaScale = mulN1N2 / sumN1N2;
        algorithmFPType meanScale  = (algorithmFPType)1 / sumN1N2;
        algorithmFPType delta      = mean - mrgMean;
#endif

#if (defined _RMIN_)
        mrgMin = fmin(min, mrgMin);
#endif
#if (defined _RMAX_)
        mrgMax = fmax(max, mrgMax);
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        mrgSum += sum;
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        mrgSum2 += sum2;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        mrgSum2Cent = mrgSum2Cent + sum2Cent + delta * delta * deltaScale;
        mrgMean     = (mrgMean * mrgVectors + mean * nVec) * meanScale;
        mrgVectors  = sumN1N2;
#endif
    }

#if (defined _RMIN_)
    gMin[itemId] = mrgMin;
#endif
#if (defined _RMAX_)
    gMax[itemId] = mrgMax;
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    gSum[itemId] = mrgSum;
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
    gSum2[itemId] = mrgSum2;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    gSum2Cent[itemId] = mrgSum2Cent;
#endif
}

/* finalize kernel */

__kernel void finalize(const algorithmFPType nObservations
#if (defined _RMIN_)
                       ,
                       __global algorithmFPType * gMin
#endif
#if (defined _RMAX_)
                       ,
                       __global algorithmFPType * gMax
#endif
#if (defined _RMEAN_)
                       ,
                       __global algorithmFPType * gSum
#endif
#if (defined _RSORM_)
                       ,
                       __global algorithmFPType * gSum2
#endif
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                       ,
                       __global algorithmFPType * gSum2Cent
#endif
#if (defined _RMEAN_)
                       ,
                       __global algorithmFPType * gMean
#endif
#if (defined _RSORM_)
                       ,
                       __global algorithmFPType * gSecondOrderRawMoment
#endif
#if (defined _RVARC_)
                       ,
                       __global algorithmFPType * gVariance
#endif
#if (defined _RSTDEV_)
                       ,
                       __global algorithmFPType * gStDev
#endif
#if (defined _RVART_)
                       ,
                       __global algorithmFPType * gVariation
#endif
)
{
    const uint tid = get_global_id(0);

#if (defined _RMEAN_)
    algorithmFPType sum = gSum[tid];
#endif
#if (defined _RSORM_)
    algorithmFPType sum2 = gSum2[tid];
#endif
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType sum2Cent = gSum2Cent[tid];
#endif
#if (defined _RMEAN_) || (defined _RVART_)
    algorithmFPType mean = sum / nObservations;
#endif
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType variance = (algorithmFPType)0;
#endif
#if (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType stDev = (algorithmFPType)0;
#endif

// common vars calculation
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    variance = sum2Cent / (nObservations - (algorithmFPType)1);
#endif
#if (defined _RSTDEV_) || (defined _RVART_)
    stDev = (algorithmFPType)sqrt(variance);
#endif

// output assignment
#if (defined _RMEAN_)
    gMean[tid] = mean;
#endif
#if (defined _RSORM_)
    gSecondOrderRawMoment[tid] = sum2 / nObservations;
#endif
#if (defined _RVARC_)
    gVariance[tid] = variance;
#endif
#if (defined _RSTDEV_)
    gStDev[tid] = stDev;
#endif
#if (defined _RVART_)
    gVariation[tid] = stDev / mean;
#endif
}
