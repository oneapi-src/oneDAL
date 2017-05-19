/* file: split_layer.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace interface1
{
/**
*  Constructs parameters of the forward split layer
*  \param[in] nOutputs   Number of outputs for forward split layer
*  \param[in] nInputs    Number of inputs for backward split layer
*/
Parameter::Parameter(size_t nOutputs, size_t nInputs) : nOutputs(nOutputs), nInputs(nInputs) {};

}// namespace interface1
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
