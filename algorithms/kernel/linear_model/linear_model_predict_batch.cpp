/* file: linear_model_predict_batch.cpp */
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
//  Implementation of the regression algorithm classes.
//--
*/

#include "algorithms/linear_model/linear_model_predict_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LM_PREDICTION_RESULT_ID);

Input::Input(size_t nElements) : regression::prediction::Input(nElements)
{}
Input::Input(const Input &other) : regression::prediction::Input(other)
{}

NumericTablePtr Input::get(NumericTableInputId id) const
{
    return regression::prediction::Input::get(regression::prediction::NumericTableInputId(id));
}

linear_model::ModelPtr Input::get(ModelInputId id) const
{
    return linear_model::Model::cast(regression::prediction::Input::get(regression::prediction::ModelInputId(id)));
}

void Input::set(NumericTableInputId id, const NumericTablePtr &value)
{
    regression::prediction::Input::set(regression::prediction::NumericTableInputId(id), value);
}

void Input::set(ModelInputId id, const linear_model::ModelPtr &value)
{
    regression::prediction::Input::set(regression::prediction::ModelInputId(id), value);
}

Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, regression::prediction::Input::check(parameter, method));

    size_t nBeta = get(data)->getNumberOfColumns() + 1;
    size_t nResponses = get(model)->getNumberOfResponses();
    return checkNumericTable(get(model)->getBeta().get(),  betaStr(), 0, 0, nBeta, nResponses);
}


Result::Result(size_t nElements) : regression::prediction::Result(nElements) {}

NumericTablePtr Result::get(ResultId id) const
{
    return regression::prediction::Result::get(regression::prediction::ResultId(id));
}

void Result::set(ResultId id, const NumericTablePtr &value)
{
    regression::prediction::Result::set(regression::prediction::ResultId(id), value);
}

Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, regression::prediction::Result::check(input, par, method));
    const Input* in = static_cast<const Input *>(input);
    size_t nResponses = in->get(model)->getNumberOfResponses();

    DAAL_CHECK_EX(get(prediction)->getNumberOfColumns() == nResponses, ErrorIncorrectNumberOfFeatures, ArgumentName, predictionStr());
    return s;
}
}
}
}
}
}
