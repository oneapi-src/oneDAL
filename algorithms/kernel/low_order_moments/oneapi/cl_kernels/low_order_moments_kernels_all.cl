/* file: low_order_moments_kernels_all.cl */
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
//++
//  Implementation of low order moments kernels.
//--
*/

#define CONCAT(n, suff) n ## suff
#define FULLNAME(n, p) CONCAT(n, p)

#define singlePassBlockProcessor FULLNAME(singlePassBlockProcessor, FNAMESUFF)
#define singlePass FULLNAME(singlePass, FNAMESUFF)
#define blockProcessor FULLNAME(blockProcessor, FNAMESUFF)
#define processBlocks FULLNAME(processBlocks, FNAMESUFF)
#define mergeBlocks FULLNAME(mergeBlocks, FNAMESUFF)
#define finalize FULLNAME(finalize, FNAMESUFF)

/* single pass kernels common */
void singlePassBlockProcessor(__global const algorithmFPType* vectors,
                              const uint nVectors,
                              const uint vectorSize
                #if (defined _ONLINE_)
                              ,const algorithmFPType nObservations
                #endif
                      #if (defined _RMIN_)
                              ,__global algorithmFPType* gMin
                      #endif
                      #if (defined _RMAX_)
                              ,__global algorithmFPType* gMax
                      #endif
                      #if (defined _RSUM_) || (defined _ONLINE_) && ((defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                              ,__global algorithmFPType* gSum
                      #endif
                      #if (defined _RSUM2_) || (defined _ONLINE_) && (defined _RSORM_)
                              ,__global algorithmFPType* gSum2
                      #endif
                      #if (defined _RSUM2C_) || (defined _ONLINE_) && ((defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                              ,__global algorithmFPType* gSum2Cent
                      #endif
                #if !(defined _ONLINE_)
                      #if (defined _RMEAN_)
                              ,__global algorithmFPType* gMean
                      #endif
                      #if (defined _RSORM_)
                              ,__global algorithmFPType* gSecondOrderRawMoment
                      #endif
                      #if (defined _RVARC_)
                              ,__global algorithmFPType* gVariance
                      #endif
                      #if (defined _RSTDEV_)
                              ,__global algorithmFPType* gStDev
                      #endif
                      #if (defined _RVART_)
                              ,__global algorithmFPType* gVariation
                      #endif
                #endif
                              ,const uint rowPartIndex,
                              const uint rowParts,
                              const uint colPartIndex,
                              const uint colParts,
                              const uint tid,
                              const uint tnum)
{
    const int colOffset = colPartIndex * tnum;
    const int x = tid + colOffset;

    if (x < nVectors)
    {
        int rowPartSize = (vectorSize + rowParts - 1) / rowParts;
        const int rowOffset = rowPartSize * rowPartIndex;

        if (rowPartSize + rowOffset > vectorSize)
        {
            rowPartSize = vectorSize - rowOffset;
        }

#if (defined _ONLINE_)
    // for online mode initial values of min/max are defined later depending on nObservations 
    #if (defined _RMIN_)
            algorithmFPType min      = (algorithmFPType)0;
    #endif
    #if (defined _RMAX_)
            algorithmFPType max      = (algorithmFPType)0;
    #endif
#else
    #if (defined _RMIN_)
            algorithmFPType min      = vectors[rowOffset * nVectors + x];
    #endif
    #if (defined _RMAX_)
            algorithmFPType max      = vectors[rowOffset * nVectors + x];
    #endif
#endif

#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType sum      = (algorithmFPType)0;  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        algorithmFPType sum2     = (algorithmFPType)0;  
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType sum2Cent = (algorithmFPType)0;  
        algorithmFPType mean     = (algorithmFPType)0;  
#endif

#if (defined _ONLINE_)
        if((algorithmFPType)0 == nObservations)
        {
    #if (defined _RMIN_)
            min      = vectors[rowOffset * nVectors + x];
    #endif
    #if (defined _RMAX_)
            max      = vectors[rowOffset * nVectors + x];
    #endif
    #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            sum      = (algorithmFPType)0;  
    #endif
    #if (defined _RSUM2_) || (defined _RSORM_)
            sum2     = (algorithmFPType)0;  
    #endif
    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            sum2Cent = (algorithmFPType)0;  
            mean     = (algorithmFPType)0;  
    #endif
        }
        else
        {
    #if (defined _RMIN_)
            min      = gMin [x * rowParts + rowPartIndex];
    #endif
    #if (defined _RMAX_)
            max      = gMax [x * rowParts + rowPartIndex];
    #endif
    #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            sum      = gSum [x * rowParts + rowPartIndex];  
    #endif
    #if (defined _RSUM2_) || (defined _RSORM_)
            sum2     = gSum2[x * rowParts + rowPartIndex];  
    #endif
    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            sum2Cent = gSum2Cent[x * rowParts + rowPartIndex];  
            mean     = sum/nObservations;  
    #endif
        }
#endif 

        for (int row = 0; row < rowPartSize; row++)
        {
            const int y = (row + rowOffset) * nVectors;
            const algorithmFPType el = vectors[y + x];
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    #if (defined _ONLINE_)
            algorithmFPType invN   = ((algorithmFPType)1) / (nObservations + (algorithmFPType)(row + 1));
    #else            
            algorithmFPType invN   = ((algorithmFPType)1) / (algorithmFPType)(row + 1);
    #endif
            algorithmFPType delta  = el - mean;
#endif

#if (defined _RMIN_)
            min       = fmin(el, min);
#endif
#if (defined _RMAX_)
            max       = fmax(el, max);
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            sum      += el; 
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
            sum2     += el * el; 
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            mean     += delta  * invN;
            sum2Cent += delta  * (el - mean);
#endif
        }

#if (defined _RMIN_)
        gMin [x * rowParts + rowPartIndex] = min;
#endif
#if (defined _RMAX_)
        gMax [x * rowParts + rowPartIndex] = max;
#endif
#if (defined _RSUM_) || (defined _ONLINE_) && ((defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
        gSum [x * rowParts + rowPartIndex] = sum;  
#endif
#if (defined _RSUM2_) || (defined _ONLINE_) && (defined _RSORM_)
        gSum2[x * rowParts + rowPartIndex] = sum2;
#endif
#if (defined _RSUM2C_) || (defined _ONLINE_) && ((defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
        gSum2Cent[x * rowParts + rowPartIndex] = sum2Cent;
#endif

#if !(defined _ONLINE_)
    // common vars calculation
    #if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType variance = sum2Cent / (rowPartSize - (algorithmFPType)1);
    #endif
    #if (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType stDev    = (algorithmFPType)sqrt(variance);
    #endif

    // output assignment
    #if (defined _RMEAN_)
        gMean[x * rowParts + rowPartIndex] = mean;
    #endif
    #if (defined _RSORM_)
        gSecondOrderRawMoment[x * rowParts + rowPartIndex] = sum2/rowPartSize;
    #endif
    #if (defined _RVARC_)
        gVariance[x * rowParts + rowPartIndex]  = variance;
    #endif
    #if (defined _RSTDEV_)
        gStDev[x * rowParts + rowPartIndex]     = stDev; 
    #endif
    #if (defined _RVART_)
        gVariation[x * rowParts + rowPartIndex] = stDev/mean;
    #endif
#endif
    }
}

__kernel void singlePass(__global const algorithmFPType* vectors,
                            const uint nVectors,
                            const uint vectorSize
                #if (defined _ONLINE_)
                              ,const algorithmFPType nObservations
                #endif
                      #if (defined _RMIN_)
                              ,__global algorithmFPType* gMin
                      #endif
                      #if (defined _RMAX_)
                              ,__global algorithmFPType* gMax
                      #endif
                      #if (defined _RSUM_) || (defined _ONLINE_) && ((defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                              ,__global algorithmFPType* gSum
                      #endif
                      #if (defined _RSUM2_) || (defined _ONLINE_) && (defined _RSORM_)
                              ,__global algorithmFPType* gSum2
                      #endif
                      #if (defined _RSUM2C_) || (defined _ONLINE_) && ((defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                              ,__global algorithmFPType* gSum2Cent
                      #endif
                #if !(defined _ONLINE_)
                      #if (defined _RMEAN_)
                              ,__global algorithmFPType* gMean
                      #endif
                      #if (defined _RSORM_)
                              ,__global algorithmFPType* gSecondOrderRawMoment
                      #endif
                      #if (defined _RVARC_)
                              ,__global algorithmFPType* gVariance
                      #endif
                      #if (defined _RSTDEV_)
                              ,__global algorithmFPType* gStDev
                      #endif
                      #if (defined _RVART_)
                              ,__global algorithmFPType* gVariation
                      #endif
                #endif
                            )
{
    const int tid  = get_local_id(0);
    const int tnum = get_local_size(0);
    const int gid  = get_group_id(0);
    const int gnum = get_num_groups(0);

    const int colParts = (nVectors + tnum - 1) / tnum;
    const int rowParts = gnum / colParts;

    const int rowPartIndex = gid / colParts;
    const int colPartIndex = gid - rowPartIndex * colParts;

    singlePassBlockProcessor(vectors, nVectors, vectorSize 
                #if (defined _ONLINE_)
                             ,nObservations
                #endif
                      #if (defined _RMIN_)
                             ,gMin
                      #endif
                      #if (defined _RMAX_)
                             ,gMax
                      #endif
                      #if (defined _RSUM_) || (defined _ONLINE_) && ((defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                             ,gSum
                      #endif
                      #if (defined _RSUM2_) || (defined _ONLINE_) && (defined _RSORM_)
                             ,gSum2
                      #endif
                      #if (defined _RSUM2C_) || (defined _ONLINE_) && ((defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                             ,gSum2Cent
                      #endif
                #if !(defined _ONLINE_)
                      #if (defined _RMEAN_)
                             ,gMean
                      #endif
                      #if (defined _RSORM_)
                             ,gSecondOrderRawMoment
                      #endif
                      #if (defined _RVARC_)
                             ,gVariance
                      #endif
                      #if (defined _RSTDEV_)
                             ,gStDev
                      #endif
                      #if (defined _RVART_)
                             ,gVariation
                      #endif
                #endif
                             ,rowPartIndex, rowParts,
                             colPartIndex, colParts,
                             tid, tnum);
}

/* common kernels for blocks processing */

void blockProcessor(__global const algorithmFPType* vectors,
                    const uint nVectors,
                    const uint vectorSize
            #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                    ,__global uint* bNVec
            #endif
            #if (defined _RMIN_)
                    ,__global algorithmFPType* bMin
            #endif
            #if (defined _RMAX_)
                    ,__global algorithmFPType* bMax
            #endif
            #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                    ,__global algorithmFPType* bSum
            #endif
            #if (defined _RSUM2_) || (defined _RSORM_)
                    ,__global algorithmFPType* bSum2
            #endif
            #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                    ,__global algorithmFPType* bSum2Cent
            #endif
                    ,const uint rowPartIndex,
                    const uint rowParts,
                    const uint colPartIndex,
                    const uint colParts,
                    const uint tid,
                    const uint tnum)
{
    const int colOffset = colPartIndex * tnum;
    const int x = tid + colOffset;

    if (x < nVectors)
    {
        int rowPartSize     = (vectorSize + rowParts - 1) / rowParts;
        const int rowOffset = rowPartSize * rowPartIndex;

        if (rowPartSize + rowOffset > vectorSize)
        {
            rowPartSize = vectorSize - rowOffset;
        }

#if (defined _RMIN_)
        algorithmFPType min      = vectors[rowOffset * nVectors + x];
#endif
#if (defined _RMAX_)
        algorithmFPType max      = vectors[rowOffset * nVectors + x];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType sum      = (algorithmFPType)0;  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        algorithmFPType sum2     = (algorithmFPType)0;  
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType sum2Cent = (algorithmFPType)0;  
        algorithmFPType mean     = (algorithmFPType)0;  
#endif

        for (int row = 0; row < rowPartSize; row++)
        {
            const int              y = (row + rowOffset) * nVectors;
            const algorithmFPType el = vectors[y + x];
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            algorithmFPType invN   = ((algorithmFPType)1) / (algorithmFPType)(row + 1);
            algorithmFPType delta  = el - mean;
#endif

#if (defined _RMIN_)
            min       = fmin(el, min);
#endif
#if (defined _RMAX_)
            max       = fmax(el, max);
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            sum      += el; 
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
            sum2     += el * el; 
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            mean     += delta  * invN;
            sum2Cent += delta  * (el - mean);
#endif
        }

#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        bNVec[x * rowParts + rowPartIndex] = (uint)rowPartSize;
#endif
#if (defined _RMIN_)
        bMin [x * rowParts + rowPartIndex] = min;
#endif
#if (defined _RMAX_)
        bMax [x * rowParts + rowPartIndex] = max;
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        bSum [x * rowParts + rowPartIndex] = sum;  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        bSum2[x * rowParts + rowPartIndex] = sum2;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        bSum2Cent[x * rowParts + rowPartIndex] = sum2Cent;
#endif
    }
}

__kernel void processBlocks(__global const algorithmFPType* vectors,
                            const    uint             nVectors,
                            const    uint             vectorSize
                    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                            ,__global uint* bNVec
                    #endif
                    #if (defined _RMIN_)
                            ,__global algorithmFPType* bMin
                    #endif
                    #if (defined _RMAX_)
                            ,__global algorithmFPType* bMax
                    #endif
                    #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                            ,__global algorithmFPType* bSum
                    #endif
                    #if (defined _RSUM2_) || (defined _RSORM_)
                            ,__global algorithmFPType* bSum2
                    #endif
                    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                            ,__global algorithmFPType* bSum2Cent
                    #endif
                            )
{
    const int tid  = get_local_id(0);
    const int tnum = get_local_size(0);
    const int gid  = get_group_id(0);
    const int gnum = get_num_groups(0);

    const int colParts = (nVectors + tnum - 1) / tnum;
    const int rowParts = gnum / colParts;

    const int rowPartIndex = gid / colParts;
    const int colPartIndex = gid - rowPartIndex * colParts;

    blockProcessor(vectors, nVectors, vectorSize
           #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                   ,bNVec
           #endif
           #if (defined _RMIN_)
                   ,bMin
           #endif
           #if (defined _RMAX_)
                   ,bMax
           #endif
           #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                   ,bSum
           #endif
           #if (defined _RSUM2_) || (defined _RSORM_)
                   ,bSum2
           #endif
           #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                   ,bSum2Cent
           #endif
                   ,rowPartIndex, rowParts,
                   colPartIndex, colParts,
                   tid, tnum);
}

/* merge blocks kernel */
__kernel void mergeBlocks(const    uint             vectorSize
                #if (defined _ONLINE_)
                              ,const algorithmFPType nObservations
                #endif
                    #if (defined _RMIN_)
                             ,__global algorithmFPType* gMin
                    #endif
                    #if (defined _RMAX_)
                             ,__global algorithmFPType* gMax
                    #endif
                    #if (defined _RSUM_) || (defined _ONLINE_) && ((defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                             ,__global algorithmFPType* gSum
                    #endif
                    #if (defined _RSUM2_) || (defined _ONLINE_) && (defined _RSORM_)
                             ,__global algorithmFPType* gSum2
                    #endif
                    #if (defined _RSUM2C_) || (defined _ONLINE_) && ((defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
                             ,__global algorithmFPType* gSum2Cent
                    #endif
                #if !(defined _ONLINE_)
                      #if (defined _RMEAN_)
                              ,__global algorithmFPType* gMean
                      #endif
                      #if (defined _RSORM_)
                              ,__global algorithmFPType* gSecondOrderRawMoment
                      #endif
                      #if (defined _RVARC_)
                              ,__global algorithmFPType* gVariance
                      #endif
                      #if (defined _RSTDEV_)
                              ,__global algorithmFPType* gStDev
                      #endif
                      #if (defined _RVART_)
                              ,__global algorithmFPType* gVariation
                      #endif
                #endif
                    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                             ,__global uint* bNVec
                    #endif
                    #if (defined _RMIN_)
                             ,__global algorithmFPType* bMin
                    #endif
                    #if (defined _RMAX_)
                             ,__global algorithmFPType* bMax
                    #endif
                    #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                             ,__global algorithmFPType* bSum
                    #endif
                    #if (defined _RSUM2_) || (defined _RSORM_)
                             ,__global algorithmFPType* bSum2
                    #endif
                    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                             ,__global algorithmFPType* bSum2Cent
                    #endif
                            )
{
#if (defined _RMIN_)
    __local algorithmFPType lMin[LOCAL_BUFFER_SIZE];
#endif
#if (defined _RMAX_)
    __local algorithmFPType lMax[LOCAL_BUFFER_SIZE];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    __local algorithmFPType lSum[LOCAL_BUFFER_SIZE];
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
    __local algorithmFPType lSum2[LOCAL_BUFFER_SIZE];
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    __local uint            lNVec[LOCAL_BUFFER_SIZE];
    __local algorithmFPType lSum2Cent[LOCAL_BUFFER_SIZE];
    __local algorithmFPType lMean[LOCAL_BUFFER_SIZE];
#endif

    const uint localSize = get_local_size(0);
    const uint globalDim = vectorSize;
    const uint localDim  = 1;
    const uint itemId    = get_local_id(0);
    const uint groupId   = get_group_id(0);

#if (defined _RMIN_)
    algorithmFPType mrgMin      = bMin[groupId*globalDim + itemId*localDim];
#endif
#if (defined _RMAX_)
    algorithmFPType mrgMax      = bMax[groupId*globalDim + itemId*localDim];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType mrgSum      = (algorithmFPType)0;
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
    algorithmFPType mrgSum2     = (algorithmFPType)0;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType mrgVectors  = (algorithmFPType)0;
    algorithmFPType mrgSum2Cent = (algorithmFPType)0;
    algorithmFPType mrgMean     = (algorithmFPType)0;
#endif

#if (defined _ONLINE_)
    if(0 == itemId && (algorithmFPType)0 != nObservations)
    {
        // item 0 in each group performs merge of previous results
    #if (defined _RMIN_)
        mrgMin      = gMin[groupId];
    #endif
    #if (defined _RMAX_)
        mrgMax      = gMax[groupId];
    #endif
    #if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        mrgSum      = gSum[groupId];
    #endif
    #if (defined _RSUM2_) || (defined _RSORM_)
        mrgSum2     = gSum2[groupId];
    #endif
    #if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        mrgVectors  = nObservations;  
        mrgSum2Cent = gSum2Cent[groupId];
        mrgMean     = mrgSum/mrgVectors;
    #endif
    }
#endif

#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    lNVec[itemId]     =  mrgVectors; 
#endif

    for(uint i = itemId; i < vectorSize; i+= localSize)
    {
        uint offset = groupId*globalDim + i*localDim;

#if (defined _RMIN_)
        algorithmFPType min      = bMin[offset];  
#endif
#if (defined _RMAX_)
        algorithmFPType max      = bMax[offset];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType sum      = bSum[offset];  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        algorithmFPType sum2     = bSum2[offset];  
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        uint            nVec     = bNVec[offset];
        algorithmFPType sum2Cent = bSum2Cent[offset];  
        algorithmFPType mean     = sum/(algorithmFPType)nVec;
        
        algorithmFPType sumN1N2        = mrgVectors  + (algorithmFPType)nVec;
        algorithmFPType mulN1N2        = mrgVectors  * (algorithmFPType)nVec;
        algorithmFPType deltaScale     = mulN1N2 / sumN1N2;
        algorithmFPType meanScale      = (algorithmFPType)1 / sumN1N2;
        algorithmFPType delta          = mean - mrgMean;
#endif

#if (defined _RMIN_)
        mrgMin      = fmin(min, mrgMin);
#endif
#if (defined _RMAX_)
        mrgMax      = fmax(max, mrgMax);
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        mrgSum     += sum;  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        mrgSum2    += sum2;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        mrgSum2Cent = mrgSum2Cent + sum2Cent + delta*delta*deltaScale;
        mrgMean     = (mrgMean * mrgVectors + mean * (algorithmFPType)nVec)* meanScale;
        mrgVectors  = sumN1N2;
#endif

#if (defined _RMIN_)
        lMin[itemId]      = mrgMin;     
#endif
#if (defined _RMAX_)
        lMax[itemId]      = mrgMax;     
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        lSum[itemId]      = mrgSum;     
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
        lSum2[itemId]     = mrgSum2;    
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        lNVec[itemId]    += nVec; 
        lSum2Cent[itemId] = mrgSum2Cent;
        lMean[itemId]     = mrgMean;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = localSize / 2; stride > 0; stride /= 2)
    {
        if (stride > itemId)
        {
            uint offset = itemId + stride;

#if (defined _RMIN_)
            algorithmFPType min      = lMin[offset];  
#endif
#if (defined _RMAX_)
            algorithmFPType max      = lMax[offset];
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            algorithmFPType sum      = lSum[offset];  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
            algorithmFPType sum2     = lSum2[offset];  
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            uint            nVec     = lNVec[offset];
            algorithmFPType sum2Cent = lSum2Cent[offset];  
            algorithmFPType mean     = lMean[offset];

            algorithmFPType sumN1N2        = mrgVectors  + (algorithmFPType) nVec;
            algorithmFPType mulN1N2        = mrgVectors  * (algorithmFPType) nVec;
            algorithmFPType deltaScale     = mulN1N2 / sumN1N2;
            algorithmFPType meanScale      = (algorithmFPType)1 / sumN1N2;
            algorithmFPType delta          = mean - mrgMean;
#endif

#if (defined _RMIN_)
            mrgMin      = fmin(min, mrgMin);
#endif
#if (defined _RMAX_)
            mrgMax      = fmax(max, mrgMax);
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            mrgSum     += sum;  
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
            mrgSum2    += sum2;
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
            mrgSum2Cent = mrgSum2Cent + sum2Cent + delta*delta*deltaScale;
            mrgMean     = (mrgMean * mrgVectors + mean * (algorithmFPType)nVec)* meanScale;
            mrgVectors  = sumN1N2;
#endif

            // item 0 collects all results in private vars
            // but all others need to store it
            if(0 < itemId)
            {
#if (defined _RMIN_)
                lMin[itemId]      = mrgMin;     
#endif
#if (defined _RMAX_)
                lMax[itemId]      = mrgMax;     
#endif
#if (defined _RSUM_) || (defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                lSum[itemId]      = mrgSum;     
#endif
#if (defined _RSUM2_) || (defined _RSORM_)
                lSum2[itemId]     = mrgSum2;    
#endif
#if (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                lNVec[itemId]    += nVec; 
                lSum2Cent[itemId] = mrgSum2Cent;
                lMean[itemId]     = mrgMean;
#endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
     
    if (0 == itemId)
    {
#if (defined _RMIN_)
        gMin[groupId]       = mrgMin;
#endif
#if (defined _RMAX_)
        gMax[groupId]       = mrgMax;
#endif
#if (defined _RSUM_) || (defined _ONLINE_) && ((defined _RMEAN_) || (defined _RSUM2C_) || (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
        gSum[groupId]       = mrgSum;  
#endif
#if (defined _RSUM2_) || (defined _ONLINE_) && (defined _RSORM_)
        gSum2[groupId]      = mrgSum2;
#endif
#if (defined _RSUM2C_) || (defined _ONLINE_) && ((defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_))
        gSum2Cent[groupId]  = mrgSum2Cent;
#endif

#if !(defined _ONLINE_)
    // common vars calculation
    #if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType mrgVariance         = mrgSum2Cent / (mrgVectors - (algorithmFPType)1);
    #endif
    #if (defined _RSTDEV_) || (defined _RVART_)
        algorithmFPType mrgStDev            = (algorithmFPType)sqrt(mrgVariance);
    #endif

    // output assignment
    #if (defined _RMEAN_)
        gMean[groupId] = mrgMean;
    #endif
    #if (defined _RSORM_)
        gSecondOrderRawMoment[groupId] = mrgSum2/mrgVectors;
    #endif
    #if (defined _RVARC_)
        gVariance[groupId]  = mrgVariance;
    #endif
    #if (defined _RSTDEV_)
        gStDev[groupId]     = mrgStDev; 
    #endif
    #if (defined _RVART_)
        gVariation[groupId] = mrgStDev/mrgMean;
    #endif
#endif
    }
}

/* finalize kernel */

__kernel void finalize(const algorithmFPType     nObservations
                      #if (defined _RMIN_)
                          ,__global algorithmFPType* gMin
                      #endif
                      #if (defined _RMAX_)
                          ,__global algorithmFPType* gMax
                      #endif
                      #if (defined _RMEAN_)
                          ,__global algorithmFPType* gSum
                      #endif
                      #if (defined _RSORM_)
                          ,__global algorithmFPType* gSum2
                      #endif
                      #if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
                          ,__global algorithmFPType* gSum2Cent
                      #endif
                      #if (defined _RMEAN_)
                          ,__global algorithmFPType* gMean
                      #endif
                      #if (defined _RSORM_)
                          ,__global algorithmFPType* gSecondOrderRawMoment
                      #endif
                      #if (defined _RVARC_)
                          ,__global algorithmFPType* gVariance
                      #endif
                      #if (defined _RSTDEV_)
                          ,__global algorithmFPType* gStDev
                      #endif
                      #if (defined _RVART_)
                          ,__global algorithmFPType* gVariation
                      #endif
                         )
{
    const uint tid  = get_global_id(0);

#if (defined _RMEAN_)
    algorithmFPType sum      = gSum [tid];  
#endif
#if (defined _RSORM_)
    algorithmFPType sum2     = gSum2[tid];  
#endif
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType sum2Cent = gSum2Cent[tid];  
#endif
#if (defined _RMEAN_) || (defined _RVART_)
    algorithmFPType mean     = sum / nObservations;
#endif
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType variance = (algorithmFPType)0;  
#endif
#if (defined _RSTDEV_) || (defined _RVART_)
    algorithmFPType stDev    = (algorithmFPType)0;  
#endif

// common vars calculation
#if (defined _RVARC_) || (defined _RSTDEV_) || (defined _RVART_)
    variance      = sum2Cent / (nObservations - (algorithmFPType)1);
#endif
#if (defined _RSTDEV_) || (defined _RVART_)
    stDev         = (algorithmFPType)sqrt(variance);
#endif

// output assignment
#if (defined _RMEAN_)
    gMean[tid] = mean;
#endif
#if (defined _RSORM_)
    gSecondOrderRawMoment[tid] = sum2 / nObservations;
#endif
#if (defined _RVARC_)
    gVariance[tid]  = variance;
#endif
#if (defined _RSTDEV_)
    gStDev[tid]     = stDev; 
#endif
#if (defined _RVART_)
    gVariation[tid] = stDev / mean;
#endif
}

