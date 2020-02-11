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

void valuesAreFinite(float * dataPtr, bool * finitenessPrt, size_t n, bool allowNaN)
{
    uint32_t * uint32Ptr = (uint32_t *)dataPtr;
    for (size_t i = 0; i < n; ++i)
    {
        // first check: all exponent bits are 1 (inf or nan)
        if (0x7f800000u == (*uint32Ptr & 0x7f800000u))
            // second check: 1 in fraction bits (nan)
            if (0x00000000u != (*uint32Ptr & 0x007fffffu) && allowNaN)
                finitenessPrt[i] = true;
            else
                finitenessPrt[i] = false;
        else
            finitenessPrt[i] = true;
    }
}

void valuesAreFinite(double * dataPtr, bool * finitenessPrt, size_t n, bool allowNaN)
{
    uint64_t * uint64Ptr = (uint64_t *)dataPtr;
    for (size_t i = 0; i < n; ++i)
    {
        // first check: all exponent bits are 1 (inf or nan)
        if (0x7ff0000000000000uLL == (*uint64Ptr & 0x7ff0000000000000uLL))
            // second check: 1 in fraction bits (nan)
            if (0x0000000000000000uLL != (*uint64Ptr & 0x000fffffffffffffuLL) && allowNaN)
                finitenessPrt[i] = true;
            else
                finitenessPrt[i] = false;
        else
            finitenessPrt[i] = true;
    }
}

template <typename DataType, daal::CpuType cpu>
DataType computeSum(size_t nDataPtrs, size_t nElementsPerPtr, DataType ** dataPtrs)
{
    DataType sum = 0;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx)
        for (size_t i = 0; i < nElementsPerPtr; ++i) sum += dataPtrs[ptrIdx][i];

    return sum;
}

template <typename DataType, daal::CpuType cpu>
bool checkFiniteness(size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, DataType ** dataPtrs, bool allowNaN)
{
    size_t optArrLen = nElements + (64 - nElements % 64);
    daal::services::internal::TArray<bool, cpu> finiteArr(optArrLen);
    bool * finite              = finiteArr.get();
    uint64_t * uint64FinitePtr = (uint64_t *)finite;
    const uint64_t ones        = 0xffffffffffffffffuLL;

    for (size_t i = 0; i < optArrLen / 64; ++i) uint64FinitePtr[i] = ones;

    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx) valuesAreFinite(dataPtrs[ptrIdx], finite, nElementsPerPtr, allowNaN);

    for (size_t i = 0; i < optArrLen / 64; ++i)
        if (uint64FinitePtr[i] != ones) return false;
    return true;
}

#if defined(__INTEL_COMPILER)

const size_t BLOCK_SIZE       = 2048;
const size_t THREADING_BORDER = 131072;

template <typename Func>
void runBlocks(bool inParallel, size_t nBlocks, Func func)
{
    if (inParallel)
        daal::threader_for(nBlocks, nBlocks, [&](size_t i) { func(i); });
    else
        for (size_t i = 0; i < nBlocks; ++i) func(i);
}

template <typename DataType>
DataType sumWithAVX512(size_t n, DataType * dataPtr)
{
    size_t nPerInstr = 64 / sizeof(DataType);
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
DataType computeSumAVX512Impl(size_t nDataPtrs, size_t nElementsPerPtr, DataType ** dataPtrs)
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
float computeSum<float, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, float ** dataPtrs)
{
    return computeSumAVX512Impl<float>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, double ** dataPtrs)
{
    return computeSumAVX512Impl<double>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

void checkFinitenessInBlocks(float ** dataPtrs, bool * finite, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                             size_t nSurplus, bool allowNaN)
{
    const size_t nPerInstr = 16;
    __m512i exp512Mask     = _mm512_set1_epi32(0x7f800000u);
    __m512i frac512Mask    = _mm512_set1_epi32(0x007fffffu);

    runBlocks(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        __m512i * ptr512i = (__m512i *)dataPtrs[ptrIdx] + start / nPerInstr;

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            __m512i res512_1   = _mm512_and_si512(exp512Mask, ptr512i[i]);
            __m512i res512_2   = _mm512_and_si512(frac512Mask, ptr512i[i]);
            uint32_t * res32_1 = (uint32_t *)&res512_1;
            uint32_t * res32_2 = (uint32_t *)&res512_2;
            for (size_t j = 0; j < nPerInstr; ++j)
            {
                if (res32_1[j] == 0x7f800000u)
                    if (res32_2[j] != 0x00000000u && allowNaN)
                        finite[start + nPerInstr * i + j] = true;
                    else
                        finite[start + nPerInstr * i + j] = false;
                else
                    finite[start + nPerInstr * i + j] = true;
            }
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        valuesAreFinite(dataPtrs[ptrIdx] + offset, finite + offset, end - offset, allowNaN);
    });
}

void checkFinitenessInBlocks(double ** dataPtrs, bool * finite, const bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                             size_t nSurplus, bool allowNaN)
{
    const size_t nPerInstr = 8;
    __m512i exp512Mask     = _mm512_set1_epi64(0x7ff0000000000000uLL);
    __m512i frac512Mask    = _mm512_set1_epi64(0x000fffffffffffffuLL);

    runBlocks(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        __m512i * ptr512i = (__m512i *)dataPtrs[ptrIdx] + start / nPerInstr;

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            __m512i res512_1   = _mm512_and_si512(exp512Mask, ptr512i[i]);
            __m512i res512_2   = _mm512_and_si512(frac512Mask, ptr512i[i]);
            uint64_t * res64_1 = (uint64_t *)&res512_1;
            uint64_t * res64_2 = (uint64_t *)&res512_2;
            for (size_t j = 0; j < nPerInstr; ++j)
            {
                if (res64_1[j] == 0x7ff0000000000000uLL)
                    if (res64_2[j] != 0x0000000000000000uLL && allowNaN)
                        finite[start + nPerInstr * i + j] = true;
                    else
                        finite[start + nPerInstr * i + j] = false;
                else
                    finite[start + nPerInstr * i + j] = true;
            }
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        valuesAreFinite(dataPtrs[ptrIdx] + offset, finite + offset, end - offset, allowNaN);
    });
}

template <typename DataType>
bool checkFinitenessAVX512Impl(size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, DataType ** dataPtrs, bool allowNaN)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;
    size_t optArrLen    = nElements + (64 - nElements % 64);
    daal::services::internal::TArray<bool, avx512> finiteArr(optArrLen);
    bool * finite              = finiteArr.get();
    uint64_t * uint64FinitePtr = (uint64_t *)finite;
    for (size_t i = 0; i < optArrLen / 64; ++i) uint64FinitePtr[i] = 0xffffffffffffffffuLL;

    checkFinitenessInBlocks(dataPtrs, finite, inParallel, nTotalBlocks, nBlocksPerPtr, nPerBlock, nSurplus, allowNaN);

    for (size_t i = 0; i < optArrLen / 64; ++i)
        if (uint64FinitePtr[i] != 0xffffffffffffffffuLL) return false;

    return true;
}

template <>
bool checkFiniteness<float, avx512>(size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, float ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX512Impl<float>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFiniteness<double, avx512>(size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, double ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX512Impl<double>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

#endif

template <typename DataType, daal::CpuType cpu>
void allValuesAreFiniteImpl(NumericTable & table, bool allowNaN, bool * finiteness)
{
    const size_t nRows     = table.getNumberOfRows();
    const size_t nColumns  = table.getNumberOfColumns();
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

    daal::services::internal::TArray<BlockDescriptor<DataType>, cpu> blockDescrArr(nDataPtrs);
    BlockDescriptor<DataType> * blockDescrPtr = blockDescrArr.get();
    daal::services::internal::TArray<DataType *, cpu> dataPtrsArr(nDataPtrs);
    DataType ** dataPtrs = dataPtrsArr.get();
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
    bool sumIsFinite = false;
    valuesAreFinite(&sum, &sumIsFinite, 1, allowNaN);

    if (sumIsFinite)
    {
        for (size_t i = 0; i < nDataPtrs; ++i)
            if (layout == NTLayout::soa)
                table.releaseBlockOfColumnValues(blockDescrPtr[i]);
            else
                table.releaseBlockOfRows(blockDescrPtr[i]);
        *finiteness = true;
        return;
    }

    // second stage: chech finiteness of all values
    bool valAreFinite = checkFiniteness<DataType, cpu>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);

    for (size_t i = 0; i < nDataPtrs; ++i)
        if (layout == NTLayout::soa)
            table.releaseBlockOfColumnValues(blockDescrPtr[i]);
        else
            table.releaseBlockOfRows(blockDescrPtr[i]);

    *finiteness = valAreFinite;
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
