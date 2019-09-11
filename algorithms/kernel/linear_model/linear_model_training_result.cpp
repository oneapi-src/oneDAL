/* file: linear_model_training_result.cpp */
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
//  Implementation of the class defining the result of the regression training algorithm
//--
*/

#include "services/daal_defines.h"
#include "algorithms/linear_model/linear_model_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

Result::Result(size_t nElements) : regression::training::Result(nElements)
{}

linear_model::ModelPtr Result::get(ResultId id) const
{
    return linear_model::Model::cast(regression::training::Result::get(regression::training::ResultId(id)));
}

void Result::set(ResultId id, const linear_model::ModelPtr &value)
{
    regression::training::Result::set(regression::training::ResultId(id), value);
}

Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, regression::training::Result::check(input, par, method));

    const Input *in = static_cast<const Input*>(input);
    const linear_model::ModelPtr model = get(training::model);
    const size_t nFeatures = in->get(data)->getNumberOfColumns();
    DAAL_CHECK_EX(model->getNumberOfFeatures() == nFeatures, ErrorIncorrectNumberOfFeatures, services::ArgumentName, modelStr())

    const size_t nBeta = nFeatures + 1;
    const size_t nResponses = in->get(dependentVariables)->getNumberOfColumns();

    DAAL_CHECK_STATUS(s, linear_model::checkModel(model.get(), *par, nBeta, nResponses));

    return s;
}

}
}
}
}
}
