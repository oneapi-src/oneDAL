/* file: lcn_layer.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of lcn calculation algorithm and types methods.
//--
*/

#include "lcn_layer_types.h"
#include "daal_strings.h"

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
    services::Status s;
    services::Collection<size_t> dims(2);
    dims[0] = 5;
    dims[1] = 5;

    kernel = data_management::HomogenTensor<float>::create(dims, data_management::Tensor::doAllocate, 0.04f, &s);
    if (!s) return;

    sumDimension = data_management::HomogenNumericTable<float>::create(1, 1, data_management::NumericTableIface::doAllocate, (float)(1), &s);
    if (!s) return;
}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    services::Status s;
    if(indices.dims[0] > 3 || indices.dims[1] > 3)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, indicesStr()));
    }
    if(sumDimension)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(sumDimension.get(), dimensionStr(), 0, 0, 1, 1));

        data_management::NumericTablePtr dimensionTable = sumDimension;

        data_management::BlockDescriptor<int> block;
        dimensionTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataInt = block.getBlockPtr();
        size_t dim = dataInt[0];

        if(dim > 1)
        {
            return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, dimensionStr()));
        }
        dimensionTable->releaseBlockOfRows(block);
    }
    return s;
}

}// namespace interface1
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
