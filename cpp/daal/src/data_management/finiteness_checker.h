/** file finiteness_checker.h */
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

#ifndef __KERNEL_DATA_MANAGEMENT_FINITENESS_CHECKER_H__
#define __KERNEL_DATA_MANAGEMENT_FINITENESS_CHECKER_H__

#include "data_management/data/numeric_table.h"

namespace daal
{
namespace data_management
{
namespace internal
{
using namespace daal::internal;

typedef daal::data_management::NumericTable::StorageLayout NTLayout;

const uint32_t floatExpMask  = 0x7f800000u;
const uint32_t floatFracMask = 0x007fffffu;
const uint32_t floatZeroBits = 0x00000000u;

const uint64_t doubleExpMask  = 0x7ff0000000000000uLL;
const uint64_t doubleFracMask = 0x000fffffffffffffuLL;
const uint64_t doubleZeroBits = 0x0000000000000000uLL;

template <typename DataType>
DataType getInf();

bool valuesAreNotFinite(const float * dataPtr, size_t n, bool allowNaN);

bool valuesAreNotFinite(const double * dataPtr, size_t n, bool allowNaN);

template <typename DataType, daal::CpuType cpu>
DataType computeSum(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs);

template <daal::CpuType cpu>
double computeSumSOA(NumericTable & table, bool & sumIsFinite, services::Status & st);

template <typename DataType, daal::CpuType cpu>
bool checkFiniteness(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN);

template <daal::CpuType cpu>
bool checkFinitenessSOA(NumericTable & table, bool allowNaN, services::Status & st);

template <typename DataType, daal::CpuType cpu>
services::Status allValuesAreFiniteImpl(NumericTable & table, bool allowNaN, bool * finiteness);

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
