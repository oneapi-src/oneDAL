/* file: implicit_als_partial_model_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the class defining the implicit als model
//--
*/

#include "implicit_als_model.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{

template<typename modelFPType>
DAAL_EXPORT PartialModel::PartialModel(const Parameter &parameter, size_t size, modelFPType dummy)
{
    const size_t nFactors = parameter.nFactors;
    _factors.reset(new data_management::HomogenNumericTable<modelFPType>(nFactors, size, data_management::NumericTableIface::doAllocate));
    data_management::HomogenNumericTable<int> *_indicesTable = new data_management::HomogenNumericTable<int>(
        1, size, data_management::NumericTableIface::doAllocate);
    _indices.reset(_indicesTable);
    int *indicesData = _indicesTable->getArray();
    const int iSize = (int)size;
    for (int i = 0; i < iSize; i++)
    {
        indicesData[i] = i;
    }
}

template<typename modelFPType>
DAAL_EXPORT PartialModel::PartialModel(const Parameter &parameter, size_t offset,
                                       data_management::NumericTablePtr indices, modelFPType dummy)
{
    const size_t nFactors = parameter.nFactors;
    data_management::BlockDescriptor<int> block;
    const size_t size = indices->getNumberOfRows();
    indices->getBlockOfRows(0, size, data_management::readOnly, block);
    const int *srcIndicesData = block.getBlockPtr();
    _factors.reset(new data_management::HomogenNumericTable<modelFPType>(
                       nFactors, size, data_management::NumericTableIface::doAllocate));
    data_management::HomogenNumericTable<int> *_indicesTable = new data_management::HomogenNumericTable<int>(
        1, size, data_management::NumericTableIface::doAllocate);
    _indices.reset(_indicesTable);
    int *dstIndicesData = _indicesTable->getArray();
    const int iOffset = (int)offset;
    for (size_t i = 0; i < size; i++)
    {
        dstIndicesData[i] = srcIndicesData[i] + iOffset;
    }
    indices->releaseBlockOfRows(block);
}

template DAAL_EXPORT PartialModel::PartialModel(const Parameter &parameter, size_t size, DAAL_FPTYPE dummy);
template DAAL_EXPORT PartialModel::PartialModel(const Parameter &parameter, size_t offset, data_management::NumericTablePtr indices, DAAL_FPTYPE dummy);

}// namespace implicit_als
}// namespace algorithms
}// namespace daal
