/** file finiteness_checker.cpp */
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

#include "data_management/data/internal/finiteness_checker.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "service/kernel/service_data_utils.h"
#include "externals/service_dispatch.h"
#include "algorithms/threading/threading.h"
#include "service_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace internal
{
typedef daal::data_management::NumericTable::StorageLayout NTLayout;

const uint32_t floatExpMask  = 0x7f800000u;
const uint32_t floatFracMask = 0x007fffffu;
const uint32_t floatZeroBits = 0x00000000u;

const uint64_t doubleExpMask  = 0x7ff0000000000000uLL;
const uint64_t doubleFracMask = 0x000fffffffffffffuLL;
const uint64_t doubleZeroBits = 0x0000000000000000uLL;

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

bool valuesAreNotFinite(const float * dataPtr, size_t n, bool allowNaN)
{
    const uint32_t * uint32Ptr = (const uint32_t *)dataPtr;

    for (size_t i = 0; i < n; ++i)
        // check: all value exponent bits are 1 (so, it's inf or nan) and it's not allowed nan
        if (floatExpMask == (uint32Ptr[i] & floatExpMask) && !(floatZeroBits != (uint32Ptr[i] & floatFracMask) && allowNaN)) return true;
    return false;
}

bool valuesAreNotFinite(const double * dataPtr, size_t n, bool allowNaN)
{
    const uint64_t * uint64Ptr = (const uint64_t *)dataPtr;

    for (size_t i = 0; i < n; ++i)
        // check: all value exponent bits are 1 (so, it's inf or nan) and it's not allowed nan
        if (doubleExpMask == (uint64Ptr[i] & doubleExpMask) && !(doubleZeroBits != (uint64Ptr[i] & doubleFracMask) && allowNaN)) return true;
    return false;
}

template <typename DataType, daal::CpuType cpu>
DataType computeSum(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    DataType sum = 0;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx)
        for (size_t i = 0; i < nElementsPerPtr; ++i) sum += dataPtrs[ptrIdx][i];

    return sum;
}

template <typename DataType, daal::CpuType cpu>
bool checkFiniteness(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    bool notFinite = false;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx) notFinite = notFinite || valuesAreNotFinite(dataPtrs[ptrIdx], nElementsPerPtr, allowNaN);

    return !notFinite;
}

#if defined(__INTEL_COMPILER)

const size_t BLOCK_SIZE       = 8192;
const size_t THREADING_BORDER = 262144;

template <typename Func>
void runBlocks(bool inParallel, size_t nBlocks, Func func)
{
    if (inParallel)
        daal::threader_for(nBlocks, nBlocks, [&](size_t i) { func(i); });
    else
        for (size_t i = 0; i < nBlocks; ++i) func(i);
}

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

    runBlocks(inParallel, nTotalBlocks, [&](size_t iBlock) {
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
    return computeSumAVX512Impl<float>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    return computeSumAVX512Impl<double>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

services::Status checkFinitenessInBlocks(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                         size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 16;
    services::internal::TArray<bool, avx512> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    runBlocks(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m512i exp512Mask  = _mm512_set1_epi32(floatExpMask);
        __m512i frac512Mask = _mm512_set1_epi32(floatFracMask);
        __m512i zero512     = _mm512_setzero_si512();

        __mmask16 notAllowNaNMask = allowNaN ? _cvtu32_mask16(0) : _cvtu32_mask16(services::internal::MaxVal<int>::get() * 2 + 1);

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

services::Status checkFinitenessInBlocks(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                         size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 8;
    services::internal::TArray<bool, avx512> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    runBlocks(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m512i exp512Mask  = _mm512_set1_epi64(doubleExpMask);
        __m512i frac512Mask = _mm512_set1_epi64(doubleFracMask);
        __m512i zero512     = _mm512_setzero_si512();

        __mmask8 notAllowNaNMask = allowNaN ? _cvtu32_mask8(0) : _cvtu32_mask8(services::internal::MaxVal<int>::get() * 2 + 1);

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

template <typename DataType>
bool checkFinitenessAVX512Impl(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    bool finiteness;
    checkFinitenessInBlocks(dataPtrs, inParallel, nTotalBlocks, nBlocksPerPtr, nPerBlock, nSurplus, allowNaN, finiteness);
    return finiteness;
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

    size_t nDataPtrs, nElementsPerPtr;
    if (layout == NTLayout::soa)
    {
        // SOA layout: pointer for each column
        nDataPtrs       = nColumns;
        nElementsPerPtr = nRows;
    }
    else
    {
        // AOS layout: one pointer for all data
        nDataPtrs       = 1;
        nElementsPerPtr = nRows * nColumns;
    }

    services::internal::TArray<BlockDescriptor<DataType>, cpu> blockDescrArr(nDataPtrs);
    BlockDescriptor<DataType> * blockDescrPtr = blockDescrArr.get();
    services::internal::TArray<const DataType *, cpu> dataPtrsArr(nDataPtrs);
    const DataType ** dataPtrs = dataPtrsArr.get();
    DAAL_CHECK_MALLOC(blockDescrPtr);
    DAAL_CHECK_MALLOC(dataPtrs);
    for (size_t i = 0; i < nDataPtrs; ++i)
    {
        if (layout == NTLayout::soa)
            table.getBlockOfColumnValues(i, 0, nRows, readOnly, blockDescrPtr[i]);
        else
            table.getBlockOfRows(0, nRows, readOnly, blockDescrPtr[i]);

        dataPtrs[i] = blockDescrPtr[i].getBlockPtr();
    }

    // first stage: compute sum of all values and check its finiteness
    DataType sum     = computeSum<DataType, cpu>(nDataPtrs, nElementsPerPtr, dataPtrs);
    bool sumIsFinite = !valuesAreNotFinite(&sum, 1, false);

    if (sumIsFinite)
    {
        for (size_t i = 0; i < nDataPtrs; ++i)
            if (layout == NTLayout::soa)
                table.releaseBlockOfColumnValues(blockDescrPtr[i]);
            else
                table.releaseBlockOfRows(blockDescrPtr[i]);
        *finiteness = true;
        return s;
    }

    // second stage: chech finiteness of all values
    bool valuesAreFinite = checkFiniteness<DataType, cpu>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
    for (size_t i = 0; i < nDataPtrs; ++i)
        if (layout == NTLayout::soa)
            table.releaseBlockOfColumnValues(blockDescrPtr[i]);
        else
            table.releaseBlockOfRows(blockDescrPtr[i]);

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
