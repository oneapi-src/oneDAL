/* file: initializer_types.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

//++
//  Implementation of initializer types.
//--


#include "neural_networks/initializers/initializer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace interface1
{

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, services::ErrorIncorrectNumberOfInputNumericTables);
    return data_management::checkTensor(get(data).get(), dataStr());
}

services::Status Result::check(const daal::algorithms::Input *input,
    const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(Argument::size() == 1, services::ErrorIncorrectNumberOfInputNumericTables);

    const Input *algInput = static_cast<const Input *>(input);
    return data_management::checkTensor(get(value).get(), valueStr(), &(algInput->get(data)->getDimensions()));
}

} // interface1
} // initializers
} // neural_networks
} // algorithms
} // daal
