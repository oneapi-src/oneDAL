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
    std::cout << "standard sum \n";
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

    #if defined(__AVX512F__)

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

template <>
double computeSumSOA<avx512>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    return computeSumSOAAVX512Impl(table, sumIsFinite, st);
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
    checkFinitenessInBlocks512(dataPtrs, inParallel, nTotalBlocks, nBlocksPerPtr, nPerBlock, nSurplus, allowNaN, finiteness);
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
