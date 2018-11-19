/* file: neural_networks_training_model.h */
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
//  Implementation of neural network.
//--
*/

#ifndef __NEURAL_NETWORK_TRAINING_MODEL_H__
#define __NEURAL_NETWORK_TRAINING_MODEL_H__

#include "services/daal_defines.h"
#include "data_management/data/tensor.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_memory.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_forward.h"
#include "algorithms/neural_networks/layers/split/split_layer_forward.h"
#include "algorithms/neural_networks/neural_networks_prediction_model.h"
#include "algorithms/neural_networks/neural_networks_training_topology.h"

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"

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
 * \brief Class representing the parameters of neural network
 */
class Parameter : public daal::algorithms::Parameter
{
public:
    /**
     * Constructs the parameters of neural network algorithm
     * \param[in] optimizationSolver_     Optimization solver used in the neural network
     * \param[in] engine_                 Engine to be used for weights and biases initialization
     */
    Parameter(const services::SharedPtr<optimization_solver::iterative_solver::Batch > &optimizationSolver_ = services::SharedPtr<optimization_solver::iterative_solver::Batch>(),
              engines::EnginePtr engine_ = engines::mt19937::Batch<DAAL_ALGORITHM_FP_TYPE>::create()) :
                                                                                                       optimizationSolver(optimizationSolver_),
                                                                                                       engine(engine_) {}

    services::SharedPtr<optimization_solver::iterative_solver::Batch>  optimizationSolver; /*!< Optimization solver used in the neural network*/
    engines::EnginePtr engine;                                                             /*!< Engine to be used for weights and biases initialization */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__MODEL"></a>
 *  \brief Class representing the model of neural network
 */
class DAAL_EXPORT Model : public neural_networks::ModelImpl
{
public:
    DECLARE_SERIALIZABLE_CAST(Model);

    using neural_networks::ModelImpl::getWeightsAndBiases;
    using neural_networks::ModelImpl::setWeightsAndBiases;

    /** \brief Constructor */
    Model();

    static services::SharedPtr<Model> create(services::Status *stat = NULL);

    /** \brief Copy constructor */
    Model(const Model &model) :
        ModelImpl(model),
        _backwardLayers(model.getBackwardLayers()),
        _storeWeightDerivativesInTable(model._storeWeightDerivativesInTable)
    {}

    /** \brief Destructor */
    virtual ~Model() {}

    /**
     * Initializes neural network
     * \param[in] sampleSize Dimensionality of the batch for the input to the first layer
     * \param[in] topology   Collection of %LayerDescriptor of every inserted layer
     * \param[in] parameter  Parameters of the training
     *
     * \return Status of computations
     */
    template<typename modelFPType>
    services::Status initialize(const services::Collection<size_t> &sampleSize, const Topology &topology,
                                const Parameter &parameter = Parameter())
    {
        using namespace layers;
        using namespace services;

        size_t nLayers = topology.size();
        Status st;
        _backwardNextLayers = SharedPtr<Collection<NextLayers> >(new Collection<NextLayers>(nLayers));
        if (!_backwardNextLayers)
        {
            st.add(services::ErrorMemoryAllocationFailed);
            return st;
        }

        for(size_t i = 0; i < nLayers; i++)
        {
            insertLayer(topology[i]);
        }

        for(int i = (int)nLayers - 1; i >= 0; i--)
        {
            size_t layerId = topology[i].index();
            const NextLayers &next = topology[i].nextLayers();
            for (size_t j = 0; j < next.size(); j++)
            {
                (*_backwardNextLayers)[next[j]].push_back(layerId);
            }
        }

        for(int i = (int)nLayers - 1; i >= 0; i--)
        {
            layers::forward::LayerIfacePtr layer = getForwardLayer(i);
            SharedPtr<split::forward::Batch<float>  > splitLayerFloat  = dynamicPointerCast<split::forward::Batch<float>,  forward::LayerIface>(layer);
            SharedPtr<split::forward::Batch<double> > splitLayerDouble = dynamicPointerCast<split::forward::Batch<double>, forward::LayerIface>(layer);
            if(splitLayerFloat.get() || splitLayerDouble.get())
            {
                const NextLayers &next = topology[i].nextLayers();
                for (size_t j = 0; j < next.size(); j++)
                {
                    layers::forward::LayerIfacePtr nextLayer = getForwardLayer(next[j]);
                    nextLayer->getLayerParameter()->allowInplaceComputation = false;
                }
            }
        }

        allocate<modelFPType>(sampleSize, parameter);

        for(size_t i = 0; i < nLayers; i++)
        {
            getForwardLayer(i)->enableResetOnCompute(false);
            getBackwardLayer(i)->enableResetOnCompute(false);
        }
        return st;
    }

    /**
     * Returns list of forward layers
     * \return          List of forward layers
     */
    const ForwardLayersPtr getForwardLayers() const
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
    const BackwardLayersPtr getBackwardLayers() const
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
        ForwardLayersPtr _predictionForwardLayers(new ForwardLayers(nLayers));
        SharedPtr<Collection<NextLayers> > _predictionNextLayers(new Collection<NextLayers>(nLayers));
        for (size_t i = 0; i < nLayers; i++)
        {
            (*_predictionNextLayers)[i] = _nextLayers->get(i);
            (*_predictionForwardLayers)[i] = ((*_forwardLayers)[i])->getLayerForPrediction();
            (*_predictionForwardLayers)[i]->getLayerParameter()->predictionStage = true;
        }

        bool storeWeightsInTable = true;
        prediction::ModelPtr predictionModel(new prediction::Model(
                                                 _predictionForwardLayers, _predictionNextLayers, (modelFPType)0.0, storeWeightsInTable));

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
     *
     * \return Status of computations
     */
    services::Status setWeightsAndBiases(size_t idx, const data_management::NumericTablePtr &table);

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
     * \DAAL_DEPRECATED
     *
     * \return Status of computations
     */
    DAAL_DEPRECATED services::Status setErrors(services::ErrorCollection &errors)
    {
        return services::Status();
    }

    /**
     * Returns the errors of the Model
     * \DAAL_DEPRECATED
     * \return   Collection of errors
     */
    DAAL_DEPRECATED const services::ErrorCollection &getErrors() const { return _errors; }

    /**
     * Allocates the buffers needed for the training using neural network
     * \param[in] sampleSize Dimensionality of the batch for the input to the first layer
     * \param[in] parameter  Parameters of the training
     *
     * \return Status of computations
     */
    template<typename modelFPType>
    services::Status allocate(const services::Collection<size_t> &sampleSize, const Parameter &parameter = Parameter())
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        services::Status s;

        if (_sampleSize.size() > 0) { _sampleSize.clear(); }
        _sampleSize = sampleSize;

        _forwardLayers->get(0)->getLayerInput()->set(forward::data,
                TensorPtr(new HomogenTensor<modelFPType>(_sampleSize, Tensor::doAllocate)));

        size_t nLayers = _forwardLayers->size();

        for (size_t i = 0; i < nLayers; i++)
        {
            layers::Parameter *lParameter = _forwardLayers->get(i)->getLayerParameter();
            initializers::Parameter *wParameter = lParameter->weightsInitializer->getParameter();
            initializers::Parameter *bParameter = lParameter->biasesInitializer->getParameter();

            s |= connectForwardLayers(i);

            if(!wParameter->engine)
            {
               wParameter->engine = parameter.engine;
            }
            if(!bParameter->engine)
            {
               bParameter->engine = parameter.engine;
            }
        }

        bool checkWeightsAndBiasesAlloc = true;
        s |= createWeightsAndBiases<modelFPType>(checkWeightsAndBiasesAlloc);
        s |= enableConditionalGradientPropagation();
        if(!s) return s;

        for (size_t i = 0; i < nLayers; i++)
        {
            forward::LayerIfacePtr forwardLayer = _forwardLayers->get(i);
            forward::Input *forwardInput = forwardLayer->getLayerInput();

            forwardLayer->getLayerResult()->setResultForBackward(forwardInput);
        }

        /* Check weights and biases derivatives allocation status before allocating the results of backward layers */
        s |= checkWeightsAndBiasesDerivativesAllocation();

        for (int i = (int)nLayers - 1; i >= 0; i--)
        {
            s |= connectBackwardLayers(i);
        }

        s |= createWeightsAndBiasesDerivatives<modelFPType>();
        if(_solverOptionalArgumentCollection.size() == 0)
        {
            if(_storeWeightsInTable) _solverOptionalArgumentCollection = DataCollection(1);
            else                     _solverOptionalArgumentCollection = DataCollection(nLayers);
        }
        return s;
    }

protected:
    /** \brief Constructor */
    Model(services::Status &st);

    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }

    void insertLayer(const layers::LayerDescriptor &layerDescriptor)
    {
        _forwardLayers->insert(layerDescriptor.index(), layerDescriptor.layer()->forwardLayer->clone());
        _backwardLayers->insert(layerDescriptor.index(), layerDescriptor.layer()->backwardLayer->clone());
        _nextLayers->insert(layerDescriptor.index(), layerDescriptor.nextLayers());
    }

    services::Status enableConditionalGradientPropagation()
    {
        using namespace services;
        using namespace layers;

        services::Status s;

        size_t nLayers = _forwardLayers->size();

        /* Array of flags for the neural network layers */
        bool *flags = (bool *)daal_malloc(nLayers * sizeof(bool));

        /* Perform depth search to disable gradient propagation in starting forward layers with weights
           and all the previous layers */
        s |= disableGradientPropagationInStartingLayers(nLayers, flags);

        /* Perform depth search to enable gradient propagation in the layers
           that follow forward layers with weights */
        s |= enableGradientPropagation(nLayers, flags);

        daal_free(flags);
        return s;
    }

    services::Status disableGradientPropagationInStartingLayers(size_t nLayers, bool *visited)
    {
        using namespace services;
        using namespace layers;

        for (size_t i = 0; i < nLayers; i++)
        {
            visited[i] = false;
        }

        Collection<size_t> stack;
        stack.push_back(0 /* ID of the first forward layer */);
        while (stack.size() > 0)
        {
            size_t layerId = stack[stack.size() - 1];
            stack.erase(stack.size() - 1);
            if (!visited[layerId])
            {
                visited[layerId] = true;

                forward::LayerIfacePtr forwardLayer = _forwardLayers->get(layerId);
                forward::Input *forwardInput = forwardLayer->getLayerInput();
                layers::Parameter *forwardParameter = forwardLayer->getLayerParameter();
                layers::Parameter *backwardParameter = _backwardLayers->get(layerId)->getLayerParameter();

                backwardParameter->propagateGradient = false;

                if (forwardInput->getWeightsSizes(forwardParameter).size() +
                    forwardInput->getBiasesSizes(forwardParameter) .size() == 0)
                {
                    /* Continue depth search for layers that do not have weights and biases */
                    const NextLayers &next = _nextLayers->get(layerId);
                    for (size_t i = 0; i < next.size(); i++)
                    {
                        stack.push_back(next[i]);
                    }
                }
            }
        }
        return services::Status();
    }

    services::Status enableGradientPropagationInSubsequentLayers(size_t startLayerId, size_t nLayers, bool *enabledPropagation)
    {
        using namespace services;
        using namespace layers;
        Collection<size_t> stack;
        const NextLayers &next = _nextLayers->get(startLayerId);
        for (size_t i = 0; i < next.size(); i++)
        {
            stack.push_back(next[i]);
        }
        while (stack.size() > 0)
        {
            size_t layerId = stack[stack.size() - 1];
            stack.erase(stack.size() - 1);
            if (!enabledPropagation[layerId])
            {
                enabledPropagation[layerId] = true;
                backward::LayerIfacePtr backwardLayer = _backwardLayers->get(layerId);
                backwardLayer->getLayerParameter()->propagateGradient = true;
                const NextLayers &next = _nextLayers->get(layerId);
                for (size_t i = 0; i < next.size(); i++)
                {
                    stack.push_back(next[i]);
                }
            }
        }
        return services::Status();
    }

    services::Status enableGradientPropagation(size_t nLayers, bool *enabledPropagation)
    {
        using namespace services;
        using namespace layers;
        Collection<size_t> stack;
        stack.push_back(0 /* ID of the first forward layer */);

        for (size_t i = 0; i < nLayers; i++)
        {
            enabledPropagation[i] = false;
        }

        while (stack.size() > 0)
        {
            size_t layerId = stack[stack.size() - 1];
            stack.erase(stack.size() - 1);
            if (!enabledPropagation[layerId])
            {
                forward::LayerIfacePtr forwardLayer = _forwardLayers->get(layerId);
                forward::Input *forwardInput = forwardLayer->getLayerInput();
                layers::Parameter *forwardParameter = forwardLayer->getLayerParameter();
                layers::Parameter *backwardParameter = _backwardLayers->get(layerId)->getLayerParameter();

                if (backwardParameter->propagateGradient == false &&
                    (forwardInput->getWeightsSizes(forwardParameter).size() +
                     forwardInput->getBiasesSizes(forwardParameter) .size()) > 0)
                {
                    enableGradientPropagationInSubsequentLayers(layerId, nLayers, enabledPropagation);
                }
                else
                {
                    const NextLayers &next = _nextLayers->get(layerId);
                    for (size_t i = 0; i < next.size(); i++)
                    {
                        stack.push_back(next[i]);
                    }
                }
            }
        }
        return services::Status();
    }

    services::Status checkWeightsAndBiasesDerivativesAllocation()
    {
        using namespace services;
        using namespace layers;

        _storeWeightDerivativesInTable = true;
        size_t nLayers = _backwardLayers->size();
        for (size_t i = 0; i < nLayers; i++)
        {
            backward::LayerIfacePtr &backwardLayer = _backwardLayers->get(i);
            if (!backwardLayer) { continue; }
            backward::ResultPtr backwardResult = backwardLayer->getLayerResult();
            /* Check if weight and bias derivatives are allocated by user */
            if (backwardResult->get(backward::weightDerivatives) || backwardResult->get(backward::biasDerivatives))
            {
                _storeWeightDerivativesInTable = false;
                break;
            }
        }
        return services::Status();
    }

    services::Status connectBackwardLayers(size_t layerId)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        forward::LayerIfacePtr &forwardLayer = _forwardLayers->get(layerId);
        backward::LayerIfacePtr &backwardLayer = _backwardLayers->get(layerId);

        if (!forwardLayer || !backwardLayer) { return services::Status(); }

        backward::Input *backwardInput = backwardLayer->getLayerInput();
        forward::ResultPtr forwardResult = forwardLayer->getLayerResult();

        backwardInput->setInputFromForward(forwardResult);
        backwardLayer->allocateResult();

        /* Don't connect backward layer to next backward layers
           if the layer does not propagate gradient */
        if (!backwardLayer->getLayerParameter()->propagateGradient) { return services::Status(); }

        backward::ResultPtr backwardResult = backwardLayer->getLayerResult();

        const NextLayers &next = _backwardNextLayers->get(layerId);
        const size_t nextLayersSize = next.size();
        for(size_t j = 0; j < nextLayersSize; j++)
        {
            size_t inputIndex = nextLayersSize - j - 1;
            _backwardLayers->get(next[j])->addInput(backwardResult, inputIndex, 0 /* index in input object of next[j] backward layer */);
        }
        return services::Status();
    }

    template<typename modelFPType>
    DAAL_EXPORT services::Status createWeightsAndBiasesDerivatives();

public:
    /**
     * Return the OptionalArgument from the neural netowrk model that stores intermediate status of solver between epochs by index
     * \param index Index in collection of required OptionalArgument
     * \return the OptionalArgument from the neural netowrk model that stores intermediate status of solver between epochs
     */
    algorithms::OptionalArgumentPtr getSolverOptionalArgument(size_t index)
    {
        return services::dynamicPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(_solverOptionalArgumentCollection[index]);
    }

    /**
     * Sets the OptionalArgument to neural netowrk model to store intermediate status of solver between epochs
     * \param solverOptionalArgument OptionalArgumentPtr to set in collection
     * \param index Index in collection of required OptionalArgument
     *
     * \return Status of computations
     */
    services::Status setSolverOptionalArgument(const algorithms::OptionalArgumentPtr& solverOptionalArgument, size_t index)
    {
        _solverOptionalArgumentCollection[index] = solverOptionalArgument;
        return services::Status();
    }

    /**
     * Return the OptionalArgument from the neural netowrk model that stores intermediate status of solver between epochs
     * \return the OptionalArgument from the neural netowrk model that stores intermediate status of solver between epochs
     */
    data_management::DataCollection getSolverOptionalArgumentCollection()
    {
        return _solverOptionalArgumentCollection;
    }

    /**
     * Sets the OptionalArgument to neural netowrk model to store intermediate status of solver between epochs
     * \param solverOptionalArgumentCollection Structure to store intermediate status of solver
     *
     * \return Status of computations
     */
    services::Status setSolverOptionalArgumentCollection(const data_management::DataCollection &solverOptionalArgumentCollection)
    {
        _solverOptionalArgumentCollection = solverOptionalArgumentCollection;
        return services::Status();
    }

private:
    data_management::DataCollection _solverOptionalArgumentCollection;
    services::Collection<size_t> _sampleSize;
    BackwardLayersPtr _backwardLayers; /*!< List of backward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _backwardNextLayers; /*!< List of edges connecting the layers in the network */
    mutable services::ErrorCollection _errors; /*!< Collection of the errors */

    bool _storeWeightDerivativesInTable;    /*!< Flag. True if weights and biases derivatives of all the layers are stored in one numeric table */
    LearnableParametersIfacePtr _weightsAndBiasesDerivatives;
};

typedef services::SharedPtr<Model> ModelPtr;
/** @} */

} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
