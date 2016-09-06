/* file: neural_networks_prediction_model.h */
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
//  Implementation of neural network prediction model.
//--
*/

#ifndef __NEURAL_NETWORK_PREDICTION_MODEL_H__
#define __NEURAL_NETWORK_PREDICTION_MODEL_H__

#include "algorithms/algorithm.h"

#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_learnable_parameters.h"
#include "algorithms/neural_networks/neural_networks_prediction_topology.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/layer_forward.h"

namespace daal
{
namespace algorithms
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODEL"></a>
 * \brief Contains classes for training and prediction using neural network
 */
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
namespace prediction
{
namespace interface1
{
/**
 * @ingroup neural_networks_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PARAMETER"></a>
 *  \brief Class representing the parameters of neural network prediction
 */
class Parameter : public daal::algorithms::Parameter
{
public:
    /**
     * Constructs the parameters of neural network prediction algorithm
     * \param[in] batchSize_                Size of the batch to be processed by the neural network
     * \param[in] allocateWeightsAndBiases_ Flag that idicates if weights and biases are allocated or not
     */
    Parameter(size_t batchSize_ = 1, bool allocateWeightsAndBiases_ = false) :
        batchSize(batchSize_), allocateWeightsAndBiases(allocateWeightsAndBiases_)
    {}

    size_t batchSize; /*!< Size of the batch to be processed by the neural network. */
    bool allocateWeightsAndBiases;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODEL"></a>
 * \brief Class Model object for the prediction stage of neural network algorithm
 */
class Model : public neural_networks::ModelImpl
{
public:
    DAAL_CAST_OPERATOR(Model);

    /** Default constructor */
    Model() {}

    /**
     * Constructs model object for the prediction stage of neural network
     * from the list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers  List of forward stages of the layers
     * \param[in] nextLayers     List of next layers for each layer with corresponding index
     */
    Model(const services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
          const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers) :
        ModelImpl(forwardLayers, nextLayers)
    {}

    /** Copy constructor */
    Model(const Model &model) : ModelImpl(model)
    {}

    /**
     * Constructs model object for the prediction stage of neural network from a collection of layer descriptors
     * \param[in] topology  Collection of layer descriptors of every inserted layer
     */
    Model(const prediction::Topology &topology) : ModelImpl()
    {
        for(size_t i = 0; i < topology.size(); i++)
        {
            insertLayer(topology[i]);
        }
    }

    /** \brief Destructor */
    virtual ~Model() {}

    /**
     * Allocates the buffers needed for the prediction using neural network
     * \param[in] dataSize         Size of the input data for the prediction
     * \param[in] parameter        Prediction model parameter
     */
    template<typename modelFPType>
    void allocate(const services::Collection<size_t> &dataSize, const daal::algorithms::Parameter *parameter = NULL)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        size_t nLayers = _forwardLayers->size();
        Parameter defaultPar;
        const Parameter *par = (parameter ? static_cast<const Parameter *>(parameter) : &defaultPar);

        services::Collection<size_t> sampleSize(dataSize);
        sampleSize[0] = par->batchSize;

        _forwardLayers->get(0)->getLayerInput()->set(forward::data,
                                                     TensorPtr(new HomogenTensor<modelFPType>(sampleSize, Tensor::doAllocate)));

        /* Clear layers' inputs if needed */
        for (size_t i = 1; i < nLayers; i++)
        {
            _forwardLayers->get(i)->getLayerInput()->eraseInputData();
        }

        for (size_t i = 0; i < nLayers; i++)
        {
            connectForwardLayers(i);
        }

        checkWeightsAndBiasesAllocation();

        _storeWeightsInTable = (_storeWeightsInTable || par->allocateWeightsAndBiases);

        createWeightsAndBiases<modelFPType>();
    }

    /**
     * Sets list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers  List of forward stages of the layers
     * \param[in] nextLayers     List of next layers for each layer with corresponding index
     */
    void setLayers(const services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
                   const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers)
    {
        _forwardLayers = forwardLayers;
        _nextLayers = nextLayers;
    }

    /**
     * Returns the list of forward stages of the layers
     * \return List of forward stages of the layers
     */
    const services::SharedPtr<neural_networks::ForwardLayers> getLayers() const
    {
        return _forwardLayers;
    }

    /**
     * Returns the forward stage of a layer with certain index in the network
     * \param[in] index  Index of the layer in the network
     * \return Forward stage of a layer with certain index in the network
     */
    const layers::forward::LayerIfacePtr getLayer(size_t index) const
    {
        return _forwardLayers->get(index);
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_PREDICTION_MODEL_ID; }
    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {}

    void insertLayer(const layers::forward::LayerDescriptor &layerDescriptor)
    {
        _forwardLayers->insert(layerDescriptor.index(), layerDescriptor.layer);
        _nextLayers->insert(layerDescriptor.index(), layerDescriptor.nextLayers);
    }
};
/** @} */

} // namespace interface1
using interface1::Model;
typedef services::SharedPtr<Model> ModelPtr;
using interface1::Parameter;
} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} //namespace daal

#endif
