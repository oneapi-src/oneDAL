/* file: layer_types.cpp */
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
//  Implementation of neural_networks layers methods.
//--
*/

#include "layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace interface1
{

/** Default constructor */
Parameter::Parameter() : predictionStage(false),
    weightsInitializer(new initializers::uniform::Batch<>()),
    biasesInitializer(new initializers::uniform::Batch<>()),
    weightsAndBiasesInitialized(false)
{}

}// namespace interface1
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
