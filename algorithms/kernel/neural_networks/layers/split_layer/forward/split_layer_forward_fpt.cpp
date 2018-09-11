/* file: split_layer_forward_fpt.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_forward_types.h"
#include "split_layer_types.h"

#include "service_mkl_tensor.h"

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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    if (!get(layers::forward::resultForBackward))
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
        const Parameter *par = static_cast<const Parameter *>(parameter);

        const size_t nOutputs = par->nOutputs;

        LayerDataPtr resultCollection = LayerDataPtr(new LayerData());

        data_management::TensorPtr dataTensor = in->get(layers::forward::data);
        internal::MklTensor<algorithmFPType> *dataMkl = dynamic_cast<internal::MklTensor<algorithmFPType>*>( dataTensor.get() );

        if (dataMkl != 0)
        {
            const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
            for(size_t i = 0; i < nOutputs; i++)
            {
                if (par->allowInplaceComputation)
                {
                    (*resultCollection)[i] = dataTensor;
                }
                else
                {
                    (*resultCollection)[i] = data_management::TensorPtr(new internal::MklTensor<algorithmFPType>(
                                                                                              dataDims, data_management::Tensor::doAllocate));
                }
            }
        }
        else
        {
            const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
            for(size_t i = 0; i < nOutputs; i++)
            {
                if (par->allowInplaceComputation)
                {
                    (*resultCollection)[i] = dataTensor;
                }
                else
                {
                    (*resultCollection)[i] = data_management::HomogenTensor<algorithmFPType>::create(dataDims, data_management::Tensor::doAllocate, &s);
                }
            }
        }
        set(layers::forward::resultForBackward, resultCollection);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
