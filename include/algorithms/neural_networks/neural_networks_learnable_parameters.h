/* file: neural_networks_learnable_parameters.h */
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
//  Implementation of neural network model.
//--
*/

#ifndef __NEURAL_NETWORKS_LEARNABLE_PARAMETERS_H__
#define __NEURAL_NETWORKS_LEARNABLE_PARAMETERS_H__

#include "algorithms/algorithm.h"
#include "algorithms/neural_networks/neural_networks_types.h"

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/homogen_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace interface1
{
/**
 * @ingroup neural_networks
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LEARNABLEPARAMETERSIFACE"></a>
 * \brief Learnable parameters for the prediction stage of neural network algorithm
 */
class LearnableParametersIface : public data_management::SerializationIface
{
public:
    virtual ~LearnableParametersIface() {}

    virtual data_management::NumericTablePtr copyToTable() const = 0;
    virtual data_management::NumericTablePtr copyToTable(size_t idx) const = 0;
    virtual services::Status copyFromTable(const data_management::NumericTablePtr &table) = 0;
    virtual services::Status copyFromTable(const data_management::NumericTablePtr &table, size_t idx) = 0;
};
typedef services::SharedPtr<LearnableParametersIface> LearnableParametersIfacePtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODEL"></a>
 * \brief Class Model object for the prediction stage of neural network algorithm
 */
class DAAL_EXPORT ModelImpl : public daal::algorithms::Model
{
public:
    virtual ~ModelImpl() {}

    /**
     * Returns list of connections between layers
     * \return          List of next layers for each layer with corresponding index
     */
    const services::SharedPtr<services::Collection<layers::NextLayers> > getNextLayers() const
    {
        return _nextLayers;
    }

    /**
     * Sets table containing all neural network weights and biases
     * \param[in] weightsAndBiases          Table containing all neural network weights and biases
     *
     * \return Status of computations
     */
    services::Status setWeightsAndBiases(const data_management::NumericTablePtr &weightsAndBiases)
    {
        return _weightsAndBiases->copyFromTable(weightsAndBiases);
    }

    /**
     * Returns table containing all neural network weights and biases
     * \return          Table containing all neural network weights and biases
     */
    const data_management::NumericTablePtr getWeightsAndBiases() const
    {
        return _weightsAndBiases->copyToTable();
    }

protected:
    ModelImpl() :
        _forwardLayers(new neural_networks::ForwardLayers),
        _nextLayers(new services::Collection<layers::NextLayers>),
        _weightsAndBiasesCreated(false), _storeWeightsInTable(false)
    {}

    ModelImpl(services::Status &st) :
        _weightsAndBiasesCreated(false), _storeWeightsInTable(false)
    {
        _forwardLayers.reset(new neural_networks::ForwardLayers);
        if (!_forwardLayers)
            st.add(services::ErrorMemoryAllocationFailed);
        _nextLayers.reset(new services::Collection<layers::NextLayers>);
        if (!_nextLayers)
            st.add(services::ErrorMemoryAllocationFailed);
    }

    /**
     * Constructs model object for the prediction stage of neural network
     * from the list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers           List of forward stages of the layers
     * \param[in] nextLayers              List of next layers for each layer with corresponding index
     * \param[in] storeWeightsInTable     Weights and biases storage status. True if weights and biases of all layers stored in one numeric table. False otherwise
     */
    ModelImpl(const neural_networks::ForwardLayersPtr &forwardLayers,
              const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers,
              bool storeWeightsInTable = false) :
        _forwardLayers(forwardLayers), _nextLayers(nextLayers),
        _weightsAndBiasesCreated(false), _storeWeightsInTable(storeWeightsInTable) {}

    ModelImpl(const neural_networks::ForwardLayersPtr &forwardLayers,
              const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers,
              bool storeWeightsInTable, services::Status &st) :
        _forwardLayers(forwardLayers), _nextLayers(nextLayers),
        _weightsAndBiasesCreated(false), _storeWeightsInTable(storeWeightsInTable) {}

    ModelImpl(const ModelImpl &model) :
        _forwardLayers(model._forwardLayers), _nextLayers(model._nextLayers),
        _storeWeightsInTable(model._storeWeightsInTable),
        _weightsAndBiasesCreated(model._weightsAndBiasesCreated)/*,
         _weightsAndBiases(model._weightsAndBiases->clone()) */   {}

    /** Copy constructor */
    ModelImpl(const ModelImpl &model, services::Status &st) :
        _forwardLayers(model._forwardLayers), _nextLayers(model._nextLayers),
        _storeWeightsInTable(model._storeWeightsInTable),
        _weightsAndBiasesCreated(model._weightsAndBiasesCreated)/*,
         _weightsAndBiases(model._weightsAndBiases->clone()) */   {}

    services::Status checkWeightsAndBiasesAllocation()
    {
        using namespace services;
        using namespace layers;

        _storeWeightsInTable = true;
        size_t nLayers = _forwardLayers->size();
        for (size_t i = 0; i < nLayers; i++)
        {
            forward::Input *forwardInput = _forwardLayers->get(i)->getLayerInput();
            /* Check if weights and biases are allocated by user */
            if (forwardInput->get(forward::weights) || forwardInput->get(forward::biases))
            {
                _storeWeightsInTable = false;
                break;
            }
        }
        return services::Status();
    }

    services::Status connectForwardLayers(size_t layerId)
    {
        using namespace services;
        using namespace layers;

        forward::LayerIfacePtr forwardLayer = _forwardLayers->get(layerId);
        forwardLayer->allocateResult();
        forward::ResultPtr forwardResult = forwardLayer->getLayerResult();
        const NextLayers &next = _nextLayers->get(layerId);
        for(size_t j = 0; j < next.size(); j++)
        {
            _forwardLayers->get(next[j])->addInput(forwardResult, j, 0 /* index in input object of next[j] forward layer */);
        }
        return services::Status();
    }

    template<typename modelFPType>
    DAAL_EXPORT services::Status createWeightsAndBiases(bool checkAllocation);

    bool _weightsAndBiasesCreated;
    bool _storeWeightsInTable;              /*!< Flag. True if weights and biases of all the layers are stored in one numeric table */

    neural_networks::ForwardLayersPtr _forwardLayers; /*!< List of forward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _nextLayers; /*!< List of edges connecting the layers in the network */
    LearnableParametersIfacePtr _weightsAndBiases;
};
/** @} */
}
using interface1::ModelImpl;
using interface1::LearnableParametersIface;
using interface1::LearnableParametersIfacePtr;
}
}
}
#endif
