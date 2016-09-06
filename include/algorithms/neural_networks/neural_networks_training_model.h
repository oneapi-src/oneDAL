/* file: neural_networks_training_model.h */
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
//  Implementation of neural network.
//--
*/

#ifndef __NEURAL_NETWORK_TRAINING_MODEL_H__
#define __NEURAL_NETWORK_TRAINING_MODEL_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_dictionary.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_forward.h"
#include "algorithms/neural_networks/neural_networks_prediction_model.h"
#include "algorithms/neural_networks/neural_networks_training_input.h"
#include "algorithms/neural_networks/neural_networks_training_topology.h"

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "algorithms/optimization_solver/sgd/sgd_batch.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
namespace training
{
namespace interface1
{
/**
 * @ingroup neural_networks
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__PARAMETER"></a>
 *  \brief Class representing the parameters of neural network
 */
class Parameter : public daal::algorithms::Parameter
{
public:
    /**
     * Constructs the parameters of neural network algorithm
     * \param[in] batchSize_                  Size of the batch to be processed by the neural network
     * \param[in] optimizationSolver_         Optimization solver used in the neural network
     */
    Parameter(size_t batchSize_ = 1,
              services::SharedPtr<optimization_solver::iterative_solver::Batch > optimizationSolver_ =
                  services::SharedPtr<optimization_solver::iterative_solver::Batch>(new optimization_solver::sgd::Batch<float>())) :
        batchSize(batchSize_), optimizationSolver(optimizationSolver_) {};

    size_t batchSize; /*!< Size of the batch to be processed by the neural network. */

    services::SharedPtr<optimization_solver::iterative_solver::Batch>  optimizationSolver; /*!< Optimization solver used in the neural network*/
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__MODEL"></a>
 *  \brief Class representing the model of neural network
 */
class DAAL_EXPORT Model : public neural_networks::ModelImpl
{
public:
    using neural_networks::ModelImpl::getWeightsAndBiases;
    using neural_networks::ModelImpl::setWeightsAndBiases;

    DAAL_CAST_OPERATOR(Model);

    /** \brief Constructor */
    Model() : _backwardLayers(new BackwardLayers()) {}

    /** \brief Copy constructor */
    Model(const Model &model) :
        ModelImpl(model),
        _backwardLayers(model.getBackwardLayers()),
        _errors(model.getErrors()) {}

    /** \brief Destructor */
    virtual ~Model() {}

    /**
     * Initializes neural network
     * \param[in] dataSize  Dimensionality of the training data
     * \param[in] topology  Collection of %LayerDescriptor of every inserted layer
     * \param[in] parameter Parameters of the training
     */
    template<typename modelFPType>
    void initialize(const services::Collection<size_t> &dataSize, const Topology &topology,
                    const Parameter *parameter)
    {
        using namespace layers;
        using namespace services;

        size_t nLayers = topology.size();
        _backwardNextLayers = SharedPtr<Collection<NextLayers> >(new Collection<NextLayers>(nLayers));
        for(size_t i = 0; i < nLayers; i++)
        {
            insertLayer(topology[i]);

            size_t layerId = topology[i].index();
            const NextLayers &next = topology[i].nextLayers;
            for (size_t j = 0; j < next.size(); j++)
            {
                (*_backwardNextLayers)[next[j]].push_back(layerId);
            }
        }
        allocate<modelFPType>(dataSize, parameter);
    }

    /**
     * Returns list of forward layers
     * \return          List of forward layers
     */
    const services::SharedPtr<ForwardLayers> getForwardLayers() const
    {
        return _forwardLayers;
    }

    /**
     * Returns the forward stage of a layer with certain index in the network
     * \param[in] index  Index of the layer in the network
     * \return Forward stage of a layer with certain index in the network
     */
    const layers::forward::LayerIfacePtr getForwardLayer(size_t index) const
    {
        return _forwardLayers->get(index);
    }

    /**
     * Returns list of backward layers
     * \return          List of backward layers
     */
    const services::SharedPtr<BackwardLayers> getBackwardLayers() const
    {
        return _backwardLayers;
    }

    /**
     * Returns the backward stage of a layer with certain index in the network
     * \param[in] index  Index of the layer in the network
     * \return Backward stage of a layer with certain index in the network
     */
    const layers::backward::LayerIfacePtr getBackwardLayer(size_t index) const
    {
        return _backwardLayers->get(index);
    }

    /**
     * Returns list of forward layers and their parameters organised in the prediction::Model
     * \return          List of forward layers and their parameters organised in the prediction::Model
     */
    template<typename modelFPType>
    const prediction::ModelPtr getPredictionModel()
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        size_t nLayers = _forwardLayers->size();

        /* Copy forward layers */
        SharedPtr<ForwardLayers> _predictionForwardLayers(new ForwardLayers(nLayers));
        SharedPtr<Collection<NextLayers> > _predictionNextLayers(new Collection<NextLayers>(nLayers));
        for (size_t i = 0; i < nLayers; i++)
        {
            (*_predictionNextLayers)[i] = _nextLayers->get(i);
            if (_nextLayers->get(i).size() > 0)
            {
                /* Here if i-th layer is not the last layer */
                (*_predictionForwardLayers)[i] = ((*_forwardLayers)[i])->clone();
            }
            else
            {
                /* Replace last loss layer with the corresponding forward layer on prediction stage */
                SharedPtr<loss::forward::Batch> lossLayer = dynamicPointerCast<loss::forward::Batch, forward::LayerIface>(
                    (*_forwardLayers)[i]);
                if(lossLayer.get())
                    (*_predictionForwardLayers)[i] = lossLayer->getLayerForPrediction();
            }
            (*_predictionForwardLayers)[i]->getLayerParameter()->predictionStage = true;
        }

        prediction::ModelPtr predictionModel(new prediction::Model(_predictionForwardLayers, _predictionNextLayers));
        prediction::Parameter parameter;
        parameter.allocateWeightsAndBiases = true;
        predictionModel->allocate<modelFPType>(_sampleSize, &parameter);
        predictionModel->setWeightsAndBiases(getWeightsAndBiases());
        return predictionModel;
    }

    /**
     * Returns weights and biases storage status
     * \return Weights and biases storage status.
     * True if weights and biases of all layers stored in one numeric table. False otherwise.
     */
    bool getWeightsAndBiasesStorageStatus() const
    {
        return _storeWeightsInTable;
    }

    /**
     * Sets table containing weights and biases of one forward layer of neural network
     * \param[in] idx   Index of the forward layer
     * \param[in] table Table containing weights and biases of one forward layer of neural network
     */
    void setWeightsAndBiases(size_t idx, const data_management::NumericTablePtr &table);

    /**
     * Returns the weights and biases of the forward layer of neural network as numeric table
     * \param[in] idx Index of the backward layer
     * \return   Weights and biases derivatives container
     */
    data_management::NumericTablePtr getWeightsAndBiases(size_t idx) const;

    /**
     * Returns the weights and biases derivatives of all backward layers of neural network as numeric table
     * \return   Weights and biases derivatives container
     */
    data_management::NumericTablePtr getWeightsAndBiasesDerivatives() const;

    /**
     * Returns the weights and biases derivatives of the backward layer of neural network as numeric table
     * \param[in] idx Index of the backward layer
     * \return   Weights and biases derivatives container
     */
    data_management::NumericTablePtr getWeightsAndBiasesDerivatives(size_t idx) const;

    /**
     * Sets the error collection to the Model
     * \param[in] errors  Collection of errors
     */
    void setErrors(services::ErrorCollection &errors) { _errors = errors; }

    /**
     * Returns the errors of the Model
     * \return   Collection of errors
     */
    const services::ErrorCollection &getErrors() const { return _errors; }

    /**
     * Allocates the buffers needed for the training using neural network
     * \param[in] dataSize         Size of the input data for the training
     * \param[in] parameter        Parameters of the training
     */
    template<typename modelFPType>
    void allocate(const services::Collection<size_t> &dataSize, const daal::algorithms::Parameter *parameter)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        const Parameter *par = static_cast<const Parameter *>(parameter);

        if (_sampleSize.size() > 0) { _sampleSize.clear(); }
        _sampleSize = dataSize;
        _sampleSize[0] = par->batchSize;

        _forwardLayers->get(0)->getLayerInput()->set(forward::data,
                                                     TensorPtr(new HomogenTensor<modelFPType>(_sampleSize, Tensor::doAllocate)));

        checkWeightsAndBiasesAllocation();

        size_t nLayers = _forwardLayers->size();
        for (size_t i = 0; i < nLayers; i++)
        {
            connectForwardLayers(i);
        }

        createWeightsAndBiases<modelFPType>();

        for (size_t i = 0; i < nLayers; i++)
        {
            forward::LayerIfacePtr forwardLayer = _forwardLayers->get(i);
            forward::Input *forwardInput = forwardLayer->getLayerInput();

            forwardLayer->getLayerResult()->setResultForBackward(forwardInput);
        }

        checkWeightsAndBiasesDerivativesAllocation();

        for (int i = (int)nLayers - 1; i >= 0; i--)
        {
            connectBackwardLayers(i);
        }

        createWeightsAndBiasesDerivatives<modelFPType>();
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_TRAINING_MODEL_ID; }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
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

    void insertLayer(const layers::LayerDescriptor &layerDescriptor)
    {
        _forwardLayers->insert(layerDescriptor.index(), layerDescriptor.layer->forwardLayer);
        _backwardLayers->insert(layerDescriptor.index(), layerDescriptor.layer->backwardLayer);
        _nextLayers->insert(layerDescriptor.index(), layerDescriptor.nextLayers);
    }

    void checkWeightsAndBiasesDerivativesAllocation()
    {
        using namespace services;
        using namespace layers;

        _storeWeightDerivativesInTable = true;
        size_t nLayers = _backwardLayers->size();
        for (size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<backward::Result> backwardResult = _backwardLayers->get(i)->getLayerResult();
            /* Check if weight and bias derivatives are allocated by user */
            if (backwardResult->get(backward::weightDerivatives) || backwardResult->get(backward::biasDerivatives))
            {
                _storeWeightDerivativesInTable = false;
                break;
            }
        }
    }

    void connectBackwardLayers(size_t layerId)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        layers::forward::LayerIfacePtr &forwardLayer = _forwardLayers->get(layerId);
        layers::backward::LayerIfacePtr &backwardLayer = _backwardLayers->get(layerId);

        layers::backward::Input *backwardInput = backwardLayer->getLayerInput();
        SharedPtr<forward::Result> forwardResult = forwardLayer->getLayerResult();

        backwardInput->setInputFromForward(forwardResult);
        backwardLayer->allocateResult();

        SharedPtr<backward::Result> backwardResult = backwardLayer->getLayerResult();

        const NextLayers &next = _backwardNextLayers->get(layerId);
        for(size_t j = 0; j < next.size(); j++)
        {
            _backwardLayers->get(next[j])->addInput(backwardResult, j, 0 /* index in input object of next[j] backward layer */);
        }
    }

    template<typename modelFPType>
    DAAL_EXPORT void createWeightsAndBiasesDerivatives();

private:
    services::Collection<size_t> _sampleSize;
    services::SharedPtr<BackwardLayers> _backwardLayers; /*!< List of backward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _backwardNextLayers; /*!< List of edges connecting the layers in the network */
    mutable services::ErrorCollection _errors; /*!< Collection of the errors */

    bool _storeWeightDerivativesInTable;    /*!< Flag. True if weights and biases derivatives of all the layers are stored in one numeric table */
    services::SharedPtr<LearnableParametersIface> _weightsAndBiasesDerivatives;
};
/** @} */

} // namespace interface1
using interface1::Parameter;
using interface1::Model;
typedef services::SharedPtr<Model> ModelPtr;

} // namespace training
} // namespace neural_networks
} // namespace algorithms
} //namespace daal
#endif
