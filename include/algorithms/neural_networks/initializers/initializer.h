/* file: initializer.h */
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
//  Implementation of neural network layer.
//--
*/

#ifndef __INITIALIZERS_H__
#define __INITIALIZERS_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/initializers/initializer_types.h"

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
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INITIALIZERIFACE"></a>
 *  \brief Class representing a neural network weights and biases initializer
 */
class InitializerIface : public daal::algorithms::Analysis<batch>
{
public:
    Input  input;   /*!< Input of the initializer */

    Parameter  *parameterBase;   /*!< Parameter of the initializer */

   InitializerIface(): parameterBase(NULL) {}

    virtual ~InitializerIface() {}
};

} // namespace interface1
using interface1::InitializerIface;
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
