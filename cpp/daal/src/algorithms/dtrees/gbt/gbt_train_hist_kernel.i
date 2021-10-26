/* file: gbt_train_hist_kernel.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of the main compute-intensive functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_SPLIT_HIST_KERNEL_I__
#define __GBT_TRAIN_SPLIT_HIST_KERNEL_I__

#if defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{
namespace hist
{
template <typename algorithmFPType, CpuType cpu>
struct Result
{
    using GHSumType    = ghSum<algorithmFPType, cpu>;
    using ImpurityType = ImpurityData<algorithmFPType, cpu>;

    size_t nUnique;
    size_t iFeature;
    GHSumType * ghSums = nullptr;
    algorithmFPType gTotal;
    algorithmFPType hTotal;

    template <typename DataType>
    void release(DataType & data)
    {
        data.GH_SUMS_BUF->singleGHSums.get(iFeature).returnBlockToStorage(ghSums);
        ghSums     = nullptr;
        isReleased = true;
    }
    int isReleased = false;
    bool isFailed  = true;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, typename ImpurityType, typename GHSumType, typename SplitType,
          typename ResultType, CpuType cpu>
class MaxImpurityDecreaseHelper
{
public:
    static void find(size_t n, size_t minObservationsInLeafNode, algorithmFPType lambda, SplitType & split, const ResultType & res,
                     DAAL_INT & idxFeatureBestSplit, bool featureUnordered,
                     SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu> & data, size_t iFeature)
    {
        if (featureUnordered)
            findCategorical(n, minObservationsInLeafNode, lambda, split, res, idxFeatureBestSplit);
        else
            findOrdered(n, minObservationsInLeafNode, lambda, split, res, idxFeatureBestSplit, data, iFeature);
    }

    static void findOrdered(size_t n, size_t minObservationsInLeafNode, algorithmFPType lambda, SplitType & split, const ResultType & res,
                            DAAL_INT & idxFeatureBestSplit, SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu> & data,
                            size_t iFeature)
    {
        const size_t nUnique = res.nUnique;
        auto * aGHSum        = res.ghSums;
        size_t nLeft         = 0;

        ImpurityType imp(res.gTotal, res.hTotal);

        ImpurityType left;
        algorithmFPType bestImpDecrease = -services::internal::MaxVal<algorithmFPType>::get();

        for (size_t i = 0; i < nUnique; ++i)
        {
            if (!aGHSum[i].n) continue;
            nLeft += aGHSum[i].n;
            if ((n - nLeft) < minObservationsInLeafNode) break;
            left.add(aGHSum[i]);
            if (nLeft < minObservationsInLeafNode) continue;

            ImpurityType right(imp, left);
            //the part of the impurity decrease dependent on split itself
            const algorithmFPType impDecrease = left.value(lambda) + right.value(lambda);
            if ((impDecrease > bestImpDecrease))
            {
                split.left          = left;
                split.nLeft         = nLeft;
                idxFeatureBestSplit = i;
                bestImpDecrease     = impDecrease;
            }
        }
        split.impurityDecrease = bestImpDecrease;
    }

    static void findCategorical(size_t n, size_t minObservationsInLeafNode, algorithmFPType lambda, SplitType & split, const ResultType & res,
                                DAAL_INT & idxFeatureBestSplit)
    {
        const size_t nUnique = res.nUnique;
        auto * aGHSum        = res.ghSums;

        ImpurityType imp(res.gTotal, res.hTotal);

        algorithmFPType bestImpDecrease = -services::internal::MaxVal<algorithmFPType>::get();

        for (size_t i = 0; i < nUnique; ++i)
        {
            if ((aGHSum[i].n < minObservationsInLeafNode) || ((n - aGHSum[i].n) < minObservationsInLeafNode)) continue;
            const ImpurityType & left = aGHSum[i];
            ImpurityType right(imp, left);
            //the part of the impurity decrease dependent on split itself
            const algorithmFPType impDecrease = left.value(lambda) + right.value(lambda);
            if (impDecrease > bestImpDecrease)
            {
                idxFeatureBestSplit = i;
                bestImpDecrease     = impDecrease;
            }
        }
        if (idxFeatureBestSplit >= 0)
        {
            split.left  = (const GHSumType &)aGHSum[idxFeatureBestSplit];
            split.nLeft = aGHSum[idxFeatureBestSplit].n;
        }

        split.impurityDecrease = bestImpDecrease;
    }
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, typename GHSumType, CpuType cpu>
class GHSumsHelper
{
public:
    static void compute(const size_t iStart, const size_t n, const RowIndexType * const indexedFeature, const RowIndexType * aIdx,
                        const RowIndexType * aSampleToSourceRow, const algorithmFPType * const pgh, GHSumType * const aGHSum,
                        algorithmFPType & gTotal, algorithmFPType & hTotal, size_t level)
    {
        if (level)
        {
            computeCommon(iStart, n, indexedFeature, aIdx, pgh, aGHSum, gTotal, hTotal);
        }
        else
        {
            if (aSampleToSourceRow)
                computeCommon(iStart, n, indexedFeature, aIdx, pgh, aGHSum, gTotal, hTotal);
            else
                computeRoot(iStart, n, indexedFeature, aIdx, pgh, aGHSum, gTotal, hTotal);
        }
    }

    static void computeCommon(const size_t iStart, const size_t n, const RowIndexType * const indexedFeature, const RowIndexType * aIdx,
                              const algorithmFPType * const pgh, GHSumType * const aGHSum, algorithmFPType & gTotal, algorithmFPType & hTotal)
    {
        aIdx = aIdx + iStart;

        for (size_t i = 0; i < n; ++i)
        {
            const RowIndexType iSample = aIdx[i];
            const RowIndexType idx     = indexedFeature[iSample];
            auto & sum                 = aGHSum[idx];
            sum.n++;
            sum.g += pgh[2 * iSample];
            sum.h += pgh[2 * iSample + 1];
            gTotal += pgh[2 * iSample];
            hTotal += pgh[2 * iSample + 1];
        }
    }

    static void computeRoot(const size_t iStart, const size_t n, const RowIndexType * const indexedFeature, const RowIndexType * aIdx,
                            const algorithmFPType * const pgh, GHSumType * const aGHSum, algorithmFPType & gTotal, algorithmFPType & hTotal)
    {
        aIdx = aIdx + iStart;

        for (size_t i = 0; i < n; ++i)
        {
            const RowIndexType idx = indexedFeature[i];
            auto & sum             = aGHSum[idx];
            sum.n++;
            sum.g += pgh[2 * i];
            sum.h += pgh[2 * i + 1];
            gTotal += pgh[2 * i];
            hTotal += pgh[2 * i + 1];
        }
    }
    static void computeDiff(const size_t nUnique, const GHSumType * const aGHSumPrev, const GHSumType * const aGHSumsOther, GHSumType * const aGHSums)
    {
        algorithmFPType * aGHSumsFP      = (algorithmFPType *)aGHSums;
        algorithmFPType * aGHSumPrevFP   = (algorithmFPType *)aGHSumPrev;
        algorithmFPType * aGHSumsOtherFP = (algorithmFPType *)aGHSumsOther;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nUnique * 4; ++i)
        {
            aGHSumsFP[i] = aGHSumPrevFP[i] - aGHSumsOtherFP[i];
        }
    }

    static void fillByZero(const size_t nUnique, GHSumType * const aGHSum)
    {
        services::internal::service_memset_seq<algorithmFPType, cpu>((algorithmFPType *)aGHSum, algorithmFPType(0), nUnique * 4);
    }
};

template <typename RowIndexType, typename BinIndexType, typename algorithmFPType, CpuType cpu>
struct ComputeGHSumByRows
{
    static void run(algorithmFPType * aGHSumFP, const BinIndexType * indexedFeature, const RowIndexType * aIdx, algorithmFPType * pgh,
                    size_t nFeatures, size_t iStart, size_t iEnd, size_t nRows, size_t * UniquesArr)
    {
        const size_t cacheLineSize       = 64; // bytes
        const size_t prefetchOffset      = 10; // heuristic, prefetch on 10 rows ahead
        const size_t elementsInCacheLine = cacheLineSize / sizeof(BinIndexType);

        const size_t noPrefetchSize              = services::internal::min<cpu, size_t>(prefetchOffset + elementsInCacheLine, nRows);
        const size_t iEndWithPrefetch            = services::internal::min<cpu, size_t>(nRows - noPrefetchSize, iEnd);
        const size_t nCacheLinesToPrefetchOneRow = nFeatures / elementsInCacheLine + !!(nFeatures % elementsInCacheLine);

        RowIndexType i = iStart;
        PRAGMA_IVDEP
        for (; i < iEndWithPrefetch; ++i)
        {
            DAAL_PREFETCH_READ_T0(pgh + 2 * aIdx[i + prefetchOffset]);
            const BinIndexType * ptr = indexedFeature + aIdx[i + prefetchOffset] * nFeatures;
            for (RowIndexType j = 0; j < nCacheLinesToPrefetchOneRow; j++) DAAL_PREFETCH_READ_T0(ptr + elementsInCacheLine * j);

            const BinIndexType * featIdx = indexedFeature + aIdx[i] * nFeatures;

            PRAGMA_IVDEP
            for (RowIndexType j = 0; j < nFeatures; j++)
            {
                const size_t idx = 4 * (UniquesArr[j] + (size_t)featIdx[j]);
                aGHSumFP[idx + 0] += pgh[2 * aIdx[i]];
                aGHSumFP[idx + 1] += pgh[2 * aIdx[i] + 1];
                aGHSumFP[idx + 2] += algorithmFPType(1);
            }
        }

        PRAGMA_IVDEP
        for (; i < iEnd; ++i)
        {
            const BinIndexType * featIdx = indexedFeature + aIdx[i] * nFeatures;

            PRAGMA_IVDEP
            for (RowIndexType j = 0; j < nFeatures; j++)
            {
                const size_t idx = 4 * (UniquesArr[j] + (size_t)featIdx[j]);
                aGHSumFP[idx + 0] += pgh[2 * aIdx[i]];
                aGHSumFP[idx + 1] += pgh[2 * aIdx[i] + 1];
                aGHSumFP[idx + 2] += algorithmFPType(1);
            }
        }
    }
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
struct MergeGHSums
{
    using GHSumType = ghSum<algorithmFPType, cpu>;

    // TODO: optimize for other compilers
    static void run(const size_t nUnique, const size_t iStart, const size_t iEnd, algorithmFPType ** results, const size_t nBlocks,
                    Result<algorithmFPType, cpu> & res)
    {
        algorithmFPType * cur = (algorithmFPType *)res.ghSums;
        algorithmFPType * ptr = results[0] + 4 * iStart;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < 4 * nUnique; i++) cur[i] = ptr[i];

        for (size_t iB = 1; iB < nBlocks; ++iB)
        {
            algorithmFPType * ptr = results[iB] + 4 * iStart;
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < 4 * nUnique; i++) cur[i] += ptr[i];
        }

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nUnique; ++i)
        {
            res.gTotal += res.ghSums[i].g;
            res.hTotal += res.ghSums[i].h;
        }
    }
};

#if defined(__INTEL_COMPILER)
    #if __CPUID__(DAAL_CPU) >= __sse42__
        #define SSE42_ALL DAAL_CPU
    #else
        #define SSE42_ALL sse42
    #endif

    #if __CPUID__(DAAL_CPU) >= __avx__
        #define AVX_ALL DAAL_CPU
    #else
        #define AVX_ALL avx
    #endif

template <typename RowIndexType, typename BinIndexType>
struct ComputeGHSumByRows<RowIndexType, BinIndexType, float, SSE42_ALL>
{
    static void run(float * aGHSumFP, const BinIndexType * indexedFeature, const RowIndexType * aIdx, float * pgh, size_t nFeatures, size_t iStart,
                    size_t iEnd, size_t nRows, size_t * UniquesArr)
    {
        const size_t cacheLineSize       = 64; // bytes
        const size_t prefetchOffset      = 10; // heuristic, prefetch on 10 rows ahead
        const size_t elementsInCacheLine = cacheLineSize / sizeof(IndexType);

        const size_t noPrefetchSize              = services::internal::min<SSE42_ALL, size_t>(prefetchOffset + elementsInCacheLine, nRows);
        const size_t iEndWithPrefetch            = services::internal::min<SSE42_ALL, size_t>(nRows - noPrefetchSize, iEnd);
        const size_t nCacheLinesToPrefetchOneRow = nFeatures / elementsInCacheLine + !!(nFeatures % elementsInCacheLine);

        __m128 adds;
        float * addsPtr = (float *)(&adds);
        addsPtr[2]      = 1.0f;
        addsPtr[3]      = 0.0f;

        RowIndexType i = iStart;
        PRAGMA_IVDEP
        for (; i < iEndWithPrefetch; ++i)
        {
            DAAL_PREFETCH_READ_T0(pgh + 2 * aIdx[i + prefetchOffset]);
            const BinIndexType * ptr = indexedFeature + aIdx[i + prefetchOffset] * nFeatures;
            for (IndexType j = 0; j < nCacheLinesToPrefetchOneRow; j++) DAAL_PREFETCH_READ_T0(ptr + elementsInCacheLine * j);

            const BinIndexType * featIdx = indexedFeature + aIdx[i] * nFeatures;
            addsPtr[0]                   = pgh[2 * aIdx[i]];
            addsPtr[1]                   = pgh[2 * aIdx[i] + 1];

            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (IndexType j = 0; j < nFeatures; j++)
            {
                const size_t idx = 4 * (UniquesArr[j] + (size_t)featIdx[j]);
                __m128 hist1     = _mm_load_ps(aGHSumFP + idx);
                __m128 newHist1  = _mm_add_ps(adds, hist1);
                _mm_store_ps(aGHSumFP + idx, newHist1);
            }
        }

        PRAGMA_IVDEP
        for (; i < iEnd; ++i)
        {
            const BinIndexType * featIdx = indexedFeature + aIdx[i] * nFeatures;
            addsPtr[0]                   = pgh[2 * aIdx[i]];
            addsPtr[1]                   = pgh[2 * aIdx[i] + 1];

            PRAGMA_IVDEP
            for (IndexType j = 0; j < nFeatures; j++)
            {
                const size_t idx = 4 * (UniquesArr[j] + (size_t)featIdx[j]);
                __m128 hist1     = _mm_load_ps(aGHSumFP + idx);
                __m128 newHist1  = _mm_add_ps(adds, hist1);
                _mm_store_ps(aGHSumFP + idx, newHist1);
            }
        }
    }
};

template <typename RowIndexType, typename BinIndexType>
struct ComputeGHSumByRows<RowIndexType, BinIndexType, double, AVX_ALL>
{
    static void run(double * aGHSumFP, const BinIndexType * indexedFeature, const RowIndexType * aIdx, double * pgh, size_t nFeatures, size_t iStart,
                    size_t iEnd, size_t nRows, size_t * UniquesArr)
    {
        const size_t cacheLineSize       = 64; // bytes
        const size_t prefetchOffset      = 10; // heuristic, prefetch on 10 rows ahead
        const size_t elementsInCacheLine = cacheLineSize / sizeof(IndexType);

        const size_t noPrefetchSize              = services::internal::min<AVX_ALL, size_t>(prefetchOffset + elementsInCacheLine, nRows);
        const size_t iEndWithPrefetch            = services::internal::min<AVX_ALL, size_t>(nRows - noPrefetchSize, iEnd);
        const size_t nCacheLinesToPrefetchOneRow = nFeatures / elementsInCacheLine + !!(nFeatures % elementsInCacheLine);

        __m256d adds;
        double * addsPtr = (double *)(&adds);
        addsPtr[2]       = 1;
        addsPtr[3]       = 0;

        RowIndexType i = iStart;
        PRAGMA_IVDEP
        for (; i < iEndWithPrefetch; ++i)
        {
            DAAL_PREFETCH_READ_T0(pgh + 2 * aIdx[i + prefetchOffset]);
            const BinIndexType * ptr = indexedFeature + aIdx[i + prefetchOffset] * nFeatures;
            for (IndexType j = 0; j < nCacheLinesToPrefetchOneRow; j++) DAAL_PREFETCH_READ_T0(ptr + elementsInCacheLine * j);

            const BinIndexType * featIdx = indexedFeature + aIdx[i] * nFeatures;
            addsPtr[0]                   = pgh[2 * aIdx[i]];
            addsPtr[1]                   = pgh[2 * aIdx[i] + 1];

            PRAGMA_IVDEP
            for (IndexType j = 0; j < nFeatures; j++)
            {
                const size_t idx = 4 * (UniquesArr[j] + (size_t)featIdx[j]);
                __m256d hist1    = _mm256_load_pd(aGHSumFP + idx);
                __m256d newHist1 = _mm256_add_pd(adds, hist1);
                _mm256_store_pd(aGHSumFP + idx, newHist1);
            }
        }

        PRAGMA_IVDEP
        for (; i < iEnd; ++i)
        {
            const BinIndexType * featIdx = indexedFeature + aIdx[i] * nFeatures;
            addsPtr[0]                   = pgh[2 * aIdx[i]];
            addsPtr[1]                   = pgh[2 * aIdx[i] + 1];

            PRAGMA_IVDEP
            for (IndexType j = 0; j < nFeatures; j++)
            {
                const size_t idx = 4 * (UniquesArr[j] + (size_t)featIdx[j]);
                __m256d hist1    = _mm256_load_pd(aGHSumFP + idx);
                __m256d newHist1 = _mm256_add_pd(adds, hist1);
                _mm256_store_pd(aGHSumFP + idx, newHist1);
            }
        }
    }
};
#endif

} /* namespace hist */
} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
