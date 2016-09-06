/* file: lcn_layer_types.cpp */
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
//  Implementation of lcn calculation algorithm and types methods.
//--
*/

#include "lcn_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
{

namespace interface1
{
/**
 *  Default constructor
 */
Parameter::Parameter() : indices(2, 3), sigmaDegenerateCasesThreshold(0.0001)
{
    services::Collection<size_t> dims(2);
    dims[0] = 5;
    dims[1] = 5;

    kernel = services::SharedPtr<data_management::Tensor>(
                    new data_management::HomogenTensor<float>(dims, data_management::Tensor::doAllocate, 0.04f));

    sumDimension = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<float>(1, 1, data_management::NumericTableIface::doAllocate, (float)(1)));
}

/**
 * Checks the correctness of the parameter
 */
void Parameter::check() const
{
    if(indices.dims[0] > 3 || indices.dims[1] > 3)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        error->setId(services::ErrorIncorrectParameter);
        error->addStringDetail(services::ArgumentName, indicesStr() );
        this->_errors->add(error);
    }
    if(sumDimension)
    {
        data_management::NumericTablePtr dimensionTable = sumDimension;

        data_management::BlockDescriptor<int> block;
        dimensionTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataInt = block.getBlockPtr();
        size_t dim = dataInt[0];

        if(dim > 1)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, dimensionStr() );
            this->_errors->add(error);
        }
        dimensionTable->releaseBlockOfRows(block);
    }
}

}// namespace interface1
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
