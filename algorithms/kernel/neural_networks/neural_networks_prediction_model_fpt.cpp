/* file: neural_networks_prediction_model_fpt.cpp */
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
//  Implementation of model of the prediction stage of neural network
//--
*/

#include "neural_networks_prediction_model.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace prediction
{

/**
 * Constructs model object for the prediction stage of neural network
 * from the list of forward stages of the layers and the list of connections between the layers.
 * And allocates storage for weights and biases of the forward layers is needed.
 * \param[in] forwardLayersForModel         List of forward stages of the layers
 * \param[in] nextLayersForModel            List of next layers for each layer with corresponding index
 * \param[in] dummy                         Data type to be used to allocate storage for weights and biases
 * \param[in] storeWeightsInTable           Flag.
 *                                          If true then the storage for weights and biases is allocated as a single piece of memory,
 *                                          otherwise weights and biases are allocated as separate tensors
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
template<typename modelFPType>
DAAL_EXPORT Model::Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                         const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                         modelFPType dummy, bool storeWeightsInTable) :
    ModelImpl(forwardLayersForModel, nextLayersForModel, storeWeightsInTable), _allocatedBatchSize(0)
{
    bool checkWeightsAndBiasesAlloc = false;
    createWeightsAndBiases<modelFPType>(checkWeightsAndBiasesAlloc);
}

template<typename modelFPType>
DAAL_EXPORT Model::Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                         const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                         modelFPType dummy, bool storeWeightsInTable, services::Status &st) :
    ModelImpl(forwardLayersForModel, nextLayersForModel, storeWeightsInTable, st), _allocatedBatchSize(0)
{
    bool checkWeightsAndBiasesAlloc = false;
    st |= createWeightsAndBiases<modelFPType>(checkWeightsAndBiasesAlloc);
}

/**
 * Constructs model object for the prediction stage of neural network
 * from the list of forward stages of the layers and the list of connections between the layers.
 * And allocates storage for weights and biases of the forward layers is needed.
 * \param[in] forwardLayersForModel  List of forward stages of the layers
 * \param[in] nextLayersForModel     List of next layers for each layer with corresponding index
 * \param[in] storeWeightsInTable    Flag.
 *                                   If true then the storage for weights and biases is allocated as a single piece of memory,
 * \param[out] stat                  Status of the model construction
 * \return Model object for the prediction stage of neural network
 */
template<typename modelFPType>
DAAL_EXPORT ModelPtr Model::create(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                                   const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                                   bool storeWeightsInTable, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, forwardLayersForModel, nextLayersForModel, (modelFPType)0.0, storeWeightsInTable);
}

template DAAL_EXPORT Model::Model(const neural_networks::ForwardLayersPtr &,
                                  const services::SharedPtr<services::Collection<layers::NextLayers> >&,
                                  DAAL_FPTYPE, bool);

template DAAL_EXPORT Model::Model(const neural_networks::ForwardLayersPtr &,
                                  const services::SharedPtr<services::Collection<layers::NextLayers> >&,
                                  DAAL_FPTYPE, bool, services::Status&);

template DAAL_EXPORT ModelPtr Model::create<DAAL_FPTYPE>(const neural_networks::ForwardLayersPtr&,
                                                         const services::SharedPtr<services::Collection<layers::NextLayers> >&,
                                                         bool, services::Status*);

} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
