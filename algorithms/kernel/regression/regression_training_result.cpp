/* file: regression_training_result.cpp */
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

/*
//++
//  Implementation of the class defining the result of the regression training algorithm
//--
*/

#include "services/daal_defines.h"
#include "algorithms/regression/regression_training_types.h"
#include "daal_strings.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

Result::Result(size_t nElements) : daal::algorithms::Result(nElements)
{}

regression::ModelPtr Result::get(ResultId id) const
{
    return staticPointerCast<regression::Model, SerializationIface>(Argument::get(id));
}

void Result::set(ResultId id, const regression::ModelPtr &value)
{
    Argument::set(id, value);
}

Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const NumericTablePtr dataTable = algInput->get(data);
    const ModelConstPtr m = get(model);

    DAAL_CHECK(m, ErrorNullModel);

    return Status();
}

}
}
}
}
}
