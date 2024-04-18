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

const size_t BLOCK_SIZE       = 8192;
const size_t THREADING_BORDER = 262144;

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

    #if (__CPUID__(DAAL_CPU) == __avx512__)

        #include "finiteness_checker_avx512_impl.i"

    #endif // __CPUID__(DAAL_CPU) == __avx512__
    #if (__CPUID__(DAAL_CPU) == __avx2__)

        #include "finiteness_checker_avx2_impl.i"

    #endif // __CPUID__(DAAL_CPU) == __avx2__

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

template services::Status allValuesAreFiniteImpl<float, DAAL_CPU>(NumericTable & table, bool allowNaN, bool * finiteness);
template services::Status allValuesAreFiniteImpl<double, DAAL_CPU>(NumericTable & table, bool allowNaN, bool * finiteness);

} // namespace internal
} // namespace data_management
} // namespace daal
