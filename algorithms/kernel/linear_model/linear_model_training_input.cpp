/* file: linear_model_training_input.cpp */
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
//  Implementation of the class defining the input objects
//  of the regression training algorithm
//--
*/

#include "algorithms/linear_model/linear_model_training_types.h"

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
Input::Input(size_t nElements) : regression::training::Input(nElements)
{}
Input::Input(const Input& other) : regression::training::Input(other)
{}

data_management::NumericTablePtr Input::get(InputId id) const
{
    return regression::training::Input::get(regression::training::InputId(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr &value)
{
    regression::training::Input::set(regression::training::InputId(id), value);
}

}
}
}
}
}
