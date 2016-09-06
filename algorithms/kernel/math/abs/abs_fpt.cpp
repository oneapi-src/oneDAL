/* file: abs_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of abs algorithm and types methods.
//--
*/

#include "abs_types.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace abs
{
namespace interface1
{
/**
 * Allocates memory to store the result of the absolute value function
 * \param[in] input  %Input object for the absolute value function
 * \param[in] par    %Parameter of the absolute value function
 * \param[in] method Computation method of the absolute value function
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

    if(algInput == 0) { this->_errors->add(services::ErrorNullInput); return; }
    if(algInput->get(data) == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
    if(algInput->get(data).get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

    size_t nFeatures     = algInput->get(data)->getNumberOfColumns();
    size_t nObservations = algInput->get(data)->getNumberOfRows();

    if(method == fastCSR)
    {
        data_management::NumericTableIface::StorageLayout layout = algInput->get(data)->getDataLayout();
        if(layout != data_management::NumericTableIface::csrArray)
        {
            this->_errors->add(services::ErrorIncorrectTypeOfInputNumericTable); return;
        }

        services::SharedPtr<data_management::CSRNumericTableIface> inputTable =
            services::dynamicPointerCast<data_management::CSRNumericTableIface, data_management::NumericTable>(algInput->get(data));

        size_t *resColIndices = 0, *resRowIndices = 0;
        algorithmFPType *resData = 0;
        services::SharedPtr<data_management::CSRNumericTable> resTable =
            services::SharedPtr<data_management::CSRNumericTable>(
                new data_management::CSRNumericTable(resData, resColIndices, resRowIndices, nFeatures, nObservations));

        size_t dataSize = inputTable->getDataSize();

        resTable->allocateDataMemory(dataSize);

        data_management::CSRBlockDescriptor<algorithmFPType> inputBlock;
        inputTable->getSparseBlock(0, nObservations, data_management::readOnly, inputBlock);

        resTable->getArrays<algorithmFPType>(&resData, &resColIndices, &resRowIndices);

        size_t *inColIndices = inputBlock.getBlockColumnIndicesPtr();

        for(size_t i = 0; i < dataSize; i++)
        {
            resColIndices[i] = inColIndices[i];
        }

        size_t *inRowIndices = inputBlock.getBlockRowIndicesPtr();
        for(size_t i = 0; i < nObservations + 1; i++)
        {
            resRowIndices[i] = inRowIndices[i];
        }
        inputTable->releaseSparseBlock(inputBlock);

        Argument::set(value, services::staticPointerCast<data_management::SerializationIface, data_management::CSRNumericTable>(resTable));
    }
    else
    {
        Argument::set(value, data_management::SerializationIfacePtr(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nObservations,
                                  data_management::NumericTable::doAllocate)));
    }
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace abs
}// namespace math
}// namespace algorithms
}// namespace daal
