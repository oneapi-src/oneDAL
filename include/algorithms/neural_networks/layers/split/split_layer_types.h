/* file: split_layer_types.h */
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
//  Implementation of the split layer
//--
*/

#ifndef __SPLIT_LAYER_TYPES_H__
#define __SPLIT_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup split Split Layer
 * \copydoc daal::algorithms::neural_networks::layers::split
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the split layer
 */
namespace split
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__METHOD"></a>
 * Computation methods for the split layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__PARAMETER"></a>
 * \brief split layer parameters
 */
class Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the forward split layer
    *  \param[in] nOutputs   Number of outputs for forward split layer
    *  \param[in] nInputs    Number of inputs for backward split layer
    */
    Parameter(size_t nOutputs = 1, size_t nInputs = 1) : nOutputs(nOutputs), nInputs(nInputs) {};

    size_t nOutputs;    /*!< Number of outputs for forward split layer*/
    size_t nInputs;    /*!< Number of inputs for backward split layer*/
};

} // namespace interface1
using interface1::Parameter;

} // namespace split
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
