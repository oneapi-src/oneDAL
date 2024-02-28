/** file finiteness_checker.cpp */
/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "data_management/data/internal/finiteness_checker.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_dispatch.h"
#include "src/threading/threading.h"
#include "service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/finiteness_checker.h"
#include <iostream>

namespace daal
{
namespace data_management
{
namespace internal
{
using namespace daal::internal;

template <typename DataType, daal::CpuType cpu>
DataType computeSum(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    std::cout << "standard sum, check test \n";
    #if defined(__AVX512F__)
    std::cout << "test\n";
    #endif

    DataType sum = 0;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx)
        for (size_t i = 0; i < nElementsPerPtr; ++i) sum += dataPtrs[ptrIdx][i];

    return sum;
}

template <daal::CpuType cpu>
double computeSumSOA(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    double sum                                  = 0;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    for (size_t i = 0; (i < nCols) && sumIsFinite; ++i)
    {
        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const float * colPtr = colBlock.get();
            sum += static_cast<double>(computeSum<float, cpu>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const double * colPtr = colBlock.get();
            sum += computeSum<double, cpu>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }
        sumIsFinite &= !valuesAreNotFinite(&sum, 1, false);
    }

    return sum;
}

template <typename DataType, daal::CpuType cpu>
bool checkFiniteness(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    bool notFinite = false;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx) notFinite = notFinite || valuesAreNotFinite(dataPtrs[ptrIdx], nElementsPerPtr, allowNaN);

    return !notFinite;
}

template <daal::CpuType cpu>
bool checkFinitenessSOA(NumericTable & table, bool allowNaN, services::Status & st)
{
    bool valuesAreFinite                        = true;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    for (size_t i = 0; (i < nCols) && valuesAreFinite; ++i)
    {
        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const float * colPtr = colBlock.get();
            valuesAreFinite &= checkFiniteness<float, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const double * colPtr = colBlock.get();
            valuesAreFinite &= checkFiniteness<double, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }
    }

    return valuesAreFinite;
}

#if defined(DAAL_INTEL_CPP_COMPILER)

template <typename DataType>
DataType getInf()
{
    DataType inf;
    if (sizeof(DataType) == 4)
        *((uint32_t *)(&inf)) = floatExpMask;
    else
        *((uint64_t *)(&inf)) = doubleExpMask;
    return inf;
}

/*
    #if defined(__AVX2__) and !defined(__AVX512F__)

template <typename DataType>
DataType sumWithAVX2(size_t n, const DataType * dataPtr)
{
    const size_t nPerInstr = 32 / sizeof(DataType);
    DataType sum;
    if (sizeof(DataType) == 4)
    {
        __m256 sums     = _mm256_set1_ps(0);
        __m256 * ptr256 = (__m256 *)dataPtr;
        for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm256_add_ps(sums, ptr256[i]);

        // AVX2 doesn't have reduce_add_ps, so finer-grained used of intrinsics necessary
        __m128 s0    = _mm256_castps256_ps128(sums);
        __m128 s1    = _mm256_extractf128_ps(sums, 1);
        s0           = _mm_add_ps(s0, s1);
        __m128 stemp = _mm_movehl_ps(s0, s0);
        __m128 sD    = _mm_add_ps(s0, stemp);
        __m128 hi    = _mm_shuffle_ps(sD, sD, 0x1);

        sum = _mm_cvtss_f32(_mm_add_ss(sD, hi));
    }
    else
    {
        __m256d sums     = _mm256_set1_pd(0);
        __m256d * ptr256 = (__m256d *)dataPtr;
        for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm256_add_pd(sums, ptr256[i]);

        // AVX2 doesn't have reduce_add_pd, so finer-grained used of intrinsics necessary
        __m128d s0    = _mm256_castpd256_pd128(sums);
        __m128d s1    = _mm256_extractf128_pd(sums, 1);
        s0            = _mm_add_pd(s0, s1);
        __m128d stemp = _mm_unpackhi_pd(s0, s0);

        sum = _mm_cvtsd_f64(_mm_add_sd(s0, stemp));
    }
    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}

template <typename DataType>
DataType computeSumAVX2Impl(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    size_t nElements    = nDataPtrs * nElementsPerPtr;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    daal::services::internal::TArray<DataType, avx2> partialSumsArr(nTotalBlocks);
    DataType * pSums = partialSumsArr.get();
    if (!pSums) return getInf<DataType>();
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) pSums[iBlock] = 0;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;

        pSums[iBlock] = sumWithAVX2<DataType>(end - start, dataPtrs[ptrIdx] + start);
    });

    return sumWithAVX2<DataType>(nTotalBlocks, pSums);
}

template <>
float computeSum<float, avx2>(size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs)
{
    std::cout << "avx2 sum \n";
    return computeSumAVX2Impl<float>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx2>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    std::cout << "avx2 sum \n";
    return computeSumAVX2Impl<double>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

double computeSumSOAAVX2Impl(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    SafeStatus safeStat;
    double sum                                  = 0;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<double, avx2, services::internal::ScalableCalloc<double, avx2> > tlsSum(1);
    daal::TlsMem<bool, avx2, services::internal::ScalableCalloc<bool, avx2> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        double * localSum     = tlsSum.local();
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localSum);
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            *localSum += static_cast<double>(computeSum<float, avx2>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localSum += computeSum<double, avx2>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }

        *localNotFinite |= valuesAreNotFinite(localSum, 1, false);
        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return 0;
    }

    if (breakFlag)
    {
        sum         = getInf<double>();
        sumIsFinite = false;
    }
    else
    {
        tlsSum.reduce([&](double * localSum) { sum += *localSum; });
        tlsNotFinite.reduce([&](bool * localNotFinite) { sumIsFinite &= !*localNotFinite; });
    }

    return sum;
}

template <>
double computeSumSOA<avx2>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    return computeSumSOAAVX2Impl(table, sumIsFinite, st);
}

services::Status checkFinitenessInBlocks256(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                            size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 8;
    services::internal::TArray<bool, avx2> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m256i exp256Mask  = _mm256_set1_epi32(floatExpMask);
        __m256i frac256Mask = _mm256_set1_epi32(floatFracMask);
        __m256i zero256     = _mm256_setzero_si256();

        __mmask16 notAllowNaNMask =
            allowNaN ? _cvtu32_mask16(0) : _cvtu32_mask16(static_cast<unsigned int>(services::internal::MaxVal<int>::get()) * 2 + 1);

        __m256i * ptr256i = (__m256i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m256i expBits  = _mm256_and_si256(exp256Mask, ptr256i[i]);
            __m256i fracBits = _mm256_and_si256(frac256Mask, ptr256i[i]);

            __mmask16 expAreOnes   = _mm256_cmpeq_epi32_mask(exp256Mask, expBits);
            __mmask16 fracAreZeros = _mm256_cmpeq_epi32_mask(zero256, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __mmask16 orMask    = _kor_mask16(fracAreZeros, notAllowNaNMask);
            __mmask16 finalMask = _kand_mask16(expAreOnes, orMask);

            if (_cvtmask16_u32(finalMask) != 0) notFinitePtr[iBlock] = true;
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

services::Status checkFinitenessInBlocks256(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                            size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 4;
    services::internal::TArray<bool, avx2> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m256i exp256Mask  = _mm256_set1_epi64x(doubleExpMask);
        __m256i frac256Mask = _mm256_set1_epi64x(doubleFracMask);
        __m256i zero256     = _mm256_setzero_si256();

        __mmask8 notAllowNaNMask =
            allowNaN ? _cvtu32_mask8(0) : _cvtu32_mask8(static_cast<unsigned int>(services::internal::MaxVal<int>::get()) * 2 + 1);

        __m256i * ptr256i = (__m256i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m256i expBits  = _mm256_and_si256(exp256Mask, ptr256i[i]);
            __m256i fracBits = _mm256_and_si256(frac256Mask, ptr256i[i]);

            __mmask8 expAreOnes   = _mm256_cmpeq_epi64_mask(exp256Mask, expBits);
            __mmask8 fracAreZeros = _mm256_cmpeq_epi64_mask(zero256, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __mmask8 orMask    = _kor_mask8(fracAreZeros, notAllowNaNMask);
            __mmask8 finalMask = _kand_mask8(expAreOnes, orMask);

            if (_cvtmask8_u32(finalMask) != 0) notFinitePtr[iBlock] = true;
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

template <typename DataType>
bool checkFinitenessAVX2Impl(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    bool finiteness;
    checkFinitenessInBlocks256(dataPtrs, inParallel, nTotalBlocks, nBlocksPerPtr, nPerBlock, nSurplus, allowNaN, finiteness);
    return finiteness;
}

template <>
bool checkFiniteness<float, avx2>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX2Impl<float>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFiniteness<double, avx2>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX2Impl<double>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

bool checkFinitenessSOAAVX2Impl(NumericTable & table, bool allowNaN, services::Status & st)
{
    SafeStatus safeStat;
    bool valuesAreFinite                        = true;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<bool, avx2, services::internal::ScalableCalloc<bool, avx2> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            *localNotFinite |= !checkFiniteness<float, avx2>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localNotFinite |= !checkFiniteness<double, avx2>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }

        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return false;
    }

    if (breakFlag)
    {
        valuesAreFinite = false;
    }
    else
    {
        tlsNotFinite.reduce([&](bool * localNotFinite) { valuesAreFinite &= !*localNotFinite; });
    }

    return valuesAreFinite;
}

template <>
bool checkFinitenessSOA<avx2>(NumericTable & table, bool allowNaN, services::Status & st)
{
    return checkFinitenessSOAAVX2Impl(table, allowNaN, st);
}
    #endif

*/    
    #if defined(__AVX512F__)

template <typename DataType>
DataType sumWithAVX512(size_t n, const DataType * dataPtr)
{
    const size_t nPerInstr = 64 / sizeof(DataType);
    DataType sum;
    if (sizeof(DataType) == 4)
    {
        __m512 sums     = _mm512_set1_ps(0);
        __m512 * ptr512 = (__m512 *)dataPtr;
        for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm512_add_ps(sums, ptr512[i]);
        sum = _mm512_reduce_add_ps(sums);
    }
    else
    {
        __m512d sums     = _mm512_set1_pd(0);
        __m512d * ptr512 = (__m512d *)dataPtr;
        for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm512_add_pd(sums, ptr512[i]);
        sum = _mm512_reduce_add_pd(sums);
    }
    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}


template <typename DataType>
DataType computeSumAVX512Impl(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    size_t nElements    = nDataPtrs * nElementsPerPtr;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    daal::services::internal::TArray<DataType, avx512> partialSumsArr(nTotalBlocks);
    DataType * pSums = partialSumsArr.get();
    if (!pSums) return getInf<DataType>();
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) pSums[iBlock] = 0;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;

        pSums[iBlock] = sumWithAVX512<DataType>(end - start, dataPtrs[ptrIdx] + start);
    });

    return sumWithAVX512<DataType>(nTotalBlocks, pSums);
}


template <>
float computeSum<float, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs)
{
    std::cout << "avx512 sum \n";
    return computeSumAVX512Impl<float>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    std::cout << "avx512 sum \n";
    return computeSumAVX512Impl<double>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

double computeSumSOAAVX512Impl(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    SafeStatus safeStat;
    double sum                                  = 0;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<double, avx512, services::internal::ScalableCalloc<double, avx512> > tlsSum(1);
    daal::TlsMem<bool, avx512, services::internal::ScalableCalloc<bool, avx512> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        double * localSum     = tlsSum.local();
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localSum);
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, avx512> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            *localSum += static_cast<double>(computeSumAVX512Impl<float>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, avx512> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localSum += computeSumAVX512Impl<double>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }

        *localNotFinite |= valuesAreNotFinite(localSum, 1, false);
        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return 0;
    }

    if (breakFlag)
    {
        sum         = getInf<double>();
        sumIsFinite = false;
    }
    else
    {
        tlsSum.reduce([&](double * localSum) { sum += *localSum; });
        tlsNotFinite.reduce([&](bool * localNotFinite) { sumIsFinite &= !*localNotFinite; });
    }

    return sum;
}

template <>
double computeSumSOA<avx512>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    return computeSumSOAAVX512Impl(table, sumIsFinite, st);
}

services::Status checkFinitenessInBlocks512(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                            size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 16;
    services::internal::TArray<bool, avx512> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m512i exp512Mask  = _mm512_set1_epi32(floatExpMask);
        __m512i frac512Mask = _mm512_set1_epi32(floatFracMask);
        __m512i zero512     = _mm512_setzero_si512();

        __mmask16 notAllowNaNMask =
            allowNaN ? _cvtu32_mask16(0) : _cvtu32_mask16(static_cast<unsigned int>(services::internal::MaxVal<int>::get()) * 2 + 1);

        __m512i * ptr512i = (__m512i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m512i expBits  = _mm512_and_si512(exp512Mask, ptr512i[i]);
            __m512i fracBits = _mm512_and_si512(frac512Mask, ptr512i[i]);

            __mmask16 expAreOnes   = _mm512_cmpeq_epi32_mask(exp512Mask, expBits);
            __mmask16 fracAreZeros = _mm512_cmpeq_epi32_mask(zero512, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __mmask16 orMask    = _kor_mask16(fracAreZeros, notAllowNaNMask);
            __mmask16 finalMask = _kand_mask16(expAreOnes, orMask);

            if (_cvtmask16_u32(finalMask) != 0) notFinitePtr[iBlock] = true;
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

services::Status checkFinitenessInBlocks512(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                            size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 8;
    services::internal::TArray<bool, avx512> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m512i exp512Mask  = _mm512_set1_epi64(doubleExpMask);
        __m512i frac512Mask = _mm512_set1_epi64(doubleFracMask);
        __m512i zero512     = _mm512_setzero_si512();

        __mmask8 notAllowNaNMask =
            allowNaN ? _cvtu32_mask8(0) : _cvtu32_mask8(static_cast<unsigned int>(services::internal::MaxVal<int>::get()) * 2 + 1);

        __m512i * ptr512i = (__m512i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m512i expBits  = _mm512_and_si512(exp512Mask, ptr512i[i]);
            __m512i fracBits = _mm512_and_si512(frac512Mask, ptr512i[i]);

            __mmask8 expAreOnes   = _mm512_cmpeq_epi64_mask(exp512Mask, expBits);
            __mmask8 fracAreZeros = _mm512_cmpeq_epi64_mask(zero512, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __mmask8 orMask    = _kor_mask8(fracAreZeros, notAllowNaNMask);
            __mmask8 finalMask = _kand_mask8(expAreOnes, orMask);

            if (_cvtmask8_u32(finalMask) != 0) notFinitePtr[iBlock] = true;
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

bool checkFinitenessSOAAVX512Impl(NumericTable & table, bool allowNaN, services::Status & st)
{
    SafeStatus safeStat;
    bool valuesAreFinite                        = true;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<bool, avx512, services::internal::ScalableCalloc<bool, avx512> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, avx512> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            *localNotFinite |= !checkFinitenessAVX512Impl<float>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, avx512> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localNotFinite |= !checkFinitenessAVX512Impl<double>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }

        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return false;
    }

    if (breakFlag)
    {
        valuesAreFinite = false;
    }
    else
    {
        tlsNotFinite.reduce([&](bool * localNotFinite) { valuesAreFinite &= !*localNotFinite; });
    }

    return valuesAreFinite;
}


template <>
bool checkFiniteness<float, avx512>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX512Impl<float>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFiniteness<double, avx512>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX512Impl<double>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFinitenessSOA<avx512>(NumericTable & table, bool allowNaN, services::Status & st)
{
    return checkFinitenessSOAAVX512Impl(table, allowNaN, st);
}


    #endif
#endif


template <typename DataType, daal::CpuType cpu>
services::Status allValuesAreFiniteImpl(NumericTable & table, bool allowNaN, bool * finiteness)
{
    services::Status s;
    const size_t nRows    = table.getNumberOfRows();
    const size_t nColumns = table.getNumberOfColumns();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nColumns);
    const size_t nElements = nRows * nColumns;
    const NTLayout layout  = table.getDataLayout();

    // first stage: compute sum of all values and check its finiteness
    double sum       = 0;
    bool sumIsFinite = true;
    std::cout << "cpu: " << (int)cpu << std::endl;

    if (layout == NTLayout::soa)
    {
        sum = computeSumSOA<cpu>(table, sumIsFinite, s);
    }
    else
    {
        ReadRows<DataType, cpu> dataBlock(table, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataBlock);
        const DataType * dataPtr = dataBlock.get();

        sum = computeSum<DataType, cpu>(1, nElements, &dataPtr);
    }

    sumIsFinite &= !valuesAreNotFinite(&sum, 1, false);

    if (sumIsFinite)
    {
        *finiteness = true;
        return s;
    }

    // second stage: check finiteness of all values
    bool valuesAreFinite = true;
    if (layout == NTLayout::soa)
    {
        valuesAreFinite = checkFinitenessSOA<cpu>(table, allowNaN, s);
    }
    else
    {
        ReadRows<DataType, cpu> dataBlock(table, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataBlock);
        const DataType * dataPtr = dataBlock.get();

        valuesAreFinite = checkFiniteness<DataType, cpu>(nElements, 1, nElements, &dataPtr, allowNaN);
    }

    *finiteness = valuesAreFinite;

    return s;
}

template <typename DataType>
DAAL_EXPORT bool allValuesAreFinite(NumericTable & table, bool allowNaN)
{
    bool finiteness = false;

#define DAAL_CHECK_FINITENESS(cpuId, ...) allValuesAreFiniteImpl<DataType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_CHECK_FINITENESS, table, allowNaN, &finiteness);

#undef DAAL_CHECK_FINITENESS

    return finiteness;
}

template DAAL_EXPORT bool allValuesAreFinite<float>(NumericTable & table, bool allowNaN);
template DAAL_EXPORT bool allValuesAreFinite<double>(NumericTable & table, bool allowNaN);


} // namespace internal
} // namespace data_management
} // namespace daal
