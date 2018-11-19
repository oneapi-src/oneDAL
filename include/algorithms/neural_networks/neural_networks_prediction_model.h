/* file: neural_networks_prediction_model.h */
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

#include "algorithms/neural_networks/layers/split/split_layer_forward.h"

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
class DAAL_EXPORT Model : public neural_networks::ModelImpl
{
public:
    DECLARE_SERIALIZABLE_CAST(Model);

    /** Default constructor */
    Model();

    /**
     * Constructs empty model for the prediction stage of neural network
     * \param[out] stat Status of the model construction
     * \return Empty model for the prediction stage of neural network
     */
    static services::SharedPtr<Model> create(services::Status *stat = NULL);

    /**
     * Constructs model object for the prediction stage of neural network
     * from the list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayersForModel  List of forward stages of the layers
     * \param[in] nextLayersForModel     List of next layers for each layer with corresponding index
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
          const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel);

    /**
     * Constructs model object for the prediction stage of neural network
     * from the list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayersForModel  List of forward stages of the layers
     * \param[in] nextLayersForModel     List of next layers for each layer with corresponding index
     * \param[out] stat                  Status of the model construction
     * \return Model object for the prediction stage of neural network
     */
    static services::SharedPtr<Model> create(
        const neural_networks::ForwardLayersPtr &forwardLayersForModel,
        const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
        services::Status *stat = NULL);

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
    DAAL_EXPORT Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                      const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                      modelFPType dummy, bool storeWeightsInTable);
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
    DAAL_EXPORT static services::SharedPtr<Model> create(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                                                         const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                                                         bool storeWeightsInTable, services::Status *stat = NULL);

    /** Copy constructor */
    Model(const Model &model);

    /**
     * Constructs model object for the prediction stage of neural network from a collection of layer descriptors
     * \param[in] topology  Collection of layer descriptors of every inserted layer
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model(const prediction::Topology &topology);

    /**
     * Constructs model object for the prediction stage of neural network from a collection of layer descriptors
     * \param[in] topology  Collection of layer descriptors of every inserted layer
     * \param[out] stat     Status of the model construction
     * \return Model object for the prediction stage of neural network
     */
    static services::SharedPtr<Model> create(const prediction::Topology &topology, services::Status *stat = NULL);

    /** \brief Destructor */
    virtual ~Model() {}

    /**
     * Allocates the buffers needed for the prediction using neural network
     * \param[in] sampleSize Dimensionality of the batch for the input to the first layer
     * \param[in] parameter  Prediction model parameter
     *
     * \return Status of computations
     */
    template<typename modelFPType>
    services::Status allocate(const services::Collection<size_t> &sampleSize, const daal::algorithms::Parameter *parameter = NULL)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        services::Status s;

        Parameter defaultParameter;
        const Parameter *par = (parameter ? static_cast<const Parameter *>(parameter) : &defaultParameter);

        if (_allocatedBatchSize == par->batchSize) { return services::Status(); }

        size_t nLayers = _forwardLayers->size();

        _forwardLayers->get(0)->getLayerInput()->set(forward::data, HomogenTensor<modelFPType>::create(sampleSize, Tensor::doAllocate, &s));

        /* Clear layers' inputs if needed */
        for (size_t i = 1; i < nLayers; i++)
        {
            _forwardLayers->get(i)->getLayerInput()->eraseInputData();
        }

        for (size_t i = 0; i < nLayers; i++)
        {
            s |= connectForwardLayers(i);
        }
        if(!s) return s;

        bool checkWeightsAndBiasesAlloc = true;
        s |= createWeightsAndBiases<modelFPType>(checkWeightsAndBiasesAlloc);

        _allocatedBatchSize = par->batchSize;

        for(size_t i = 0; i < nLayers; i++)
        {
            getLayer(i)->enableResetOnCompute(false);
        }

        for(size_t i = 0; i < nLayers; i++)
        {
            layers::forward::LayerIfacePtr layer = _forwardLayers->get(i);
            SharedPtr<split::forward::Batch<float>  > splitLayerFloat  = dynamicPointerCast<split::forward::Batch<float>,  forward::LayerIface>(layer);
            SharedPtr<split::forward::Batch<double> > splitLayerDouble = dynamicPointerCast<split::forward::Batch<double>, forward::LayerIface>(layer);
            if(splitLayerFloat.get() || splitLayerDouble.get())
            {
                const NextLayers &next = _nextLayers->get(i);
                for (size_t j = 0; j < next.size(); j++)
                {
                    layers::forward::LayerIfacePtr nextLayer = _forwardLayers->get(next[j]);
                    nextLayer->getLayerParameter()->allowInplaceComputation = false;
                }
            }
        }
        return s;
    }

    /**
     * Sets list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers  List of forward stages of the layers
     * \param[in] nextLayers     List of next layers for each layer with corresponding index
     *
     * \return Status of computations
     */
    services::Status setLayers(const neural_networks::ForwardLayersPtr &forwardLayers,
                   const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers)
    {
        _forwardLayers = forwardLayers;
        _nextLayers = nextLayers;
        return services::Status();
    }

    /**
     * Returns the list of forward stages of the layers
     * \return List of forward stages of the layers
     */
    const neural_networks::ForwardLayersPtr getLayers() const
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

protected:
    size_t _allocatedBatchSize;  /** Batch size that was used during the model allocation */

    Model(services::Status &st);

    Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
          const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
          services::Status &st);

    template<typename modelFPType>
    DAAL_EXPORT Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                      const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                      modelFPType dummy, bool storeWeightsInTable, services::Status &st);

    Model(const prediction::Topology &topology, services::Status &st);

    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }

    services::Status insertLayer(const layers::forward::LayerDescriptor &layerDescriptor)
    {
        layers::forward::LayerIfacePtr forwardLayer = layerDescriptor.layer()->clone();
        _forwardLayers->insert(layerDescriptor.index(), forwardLayer);
        _nextLayers->insert(layerDescriptor.index(), layerDescriptor.nextLayers());

        /* Set prediction stage parameter */
        forwardLayer->getLayerParameter()->predictionStage = true;
        return services::Status();
    }
};

typedef services::SharedPtr<Model> ModelPtr;
/** @} */

} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::Parameter;
} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} //namespace daal

#endif
