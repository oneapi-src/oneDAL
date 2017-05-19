/* file: spatial_average_pooling2d_layer_forward_fpt.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_average_pooling2d_layer_types.h"
#include "spatial_average_pooling2d_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_average_pooling2d
{
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the forward spatial pyramid average 2D pooling layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of the forward spatial pyramid average 2D pooling layer
 * \param[in] method Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, spatial_pooling2d::forward::Result::allocate<algorithmFPType>(input, parameter, method));
    const Input *in = static_cast<const Input *>(input);

    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        const services::Collection<size_t> &dataDims = in->get(layers::forward::data)->getDimensions();
        set(auxInputDimensions, createAuxInputDimensions(dataDims));
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace spatial_average_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
