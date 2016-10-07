/* file: split_layer_forward_fpt.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_forward_types.h"
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
namespace forward
{
namespace interface1
{
/**
* Allocates memory to store the result of the forward split layer
* \param[in] input        Pointer to an object containing the input data
* \param[in] parameter    %Parameter of the algorithm
* \param[in] method       Computation method for the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    if (!get(layers::forward::resultForBackward))
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
        const Parameter *par = static_cast<const Parameter *>(parameter);

        size_t nOutputs = par->nOutputs;

        services::SharedPtr<LayerData> resultCollection = services::SharedPtr<LayerData>(new LayerData());

        services::SharedPtr<data_management::Tensor> dataTensor = in->get(layers::forward::data);
        const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
        for(size_t i = 0; i < nOutputs; i++)
        {
            (*resultCollection)[i] = services::SharedPtr<data_management::Tensor>(new data_management::HomogenTensor<algorithmFPType>(
                                                                                      dataDims, data_management::Tensor::doAllocate));
        }
        set(layers::forward::resultForBackward, resultCollection);
    }
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
