/* file: neural_networks_learnable_parameters.h */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LEARNABLEPARAMETERSIFACE"></a>
 * \brief Learnable parameters for the prediction stage of neural network algorithm
 */
class LearnableParametersIface : public data_management::SerializationIface
{
public:
    virtual ~LearnableParametersIface() {}

    virtual data_management::NumericTablePtr copyToTable() const = 0;
    virtual data_management::NumericTablePtr copyToTable(size_t idx) const = 0;
    virtual void copyFromTable(const data_management::NumericTablePtr &table) = 0;
    virtual void copyFromTable(const data_management::NumericTablePtr &table, size_t idx) = 0;
};

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
     */
    void setWeightsAndBiases(const data_management::NumericTablePtr &weightsAndBiases)
    {
        _weightsAndBiases->copyFromTable(weightsAndBiases);
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
        _nextLayers(new services::Collection<layers::NextLayers>)
    {}

    /**
     * Constructs model object for the prediction stage of neural network
     * from the list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers  List of forward stages of the layers
     * \param[in] nextLayers     List of next layers for each layer with corresponding index
     */
    ModelImpl(const services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
              const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers) :
        _forwardLayers(forwardLayers), _nextLayers(nextLayers) {}

    /** Copy constructor */
    ModelImpl(const ModelImpl &model) :
        _forwardLayers(model._forwardLayers), _nextLayers(model._nextLayers)/*,
         _weightsAndBiases(model._weightsAndBiases->clone()) */   {}

    void checkWeightsAndBiasesAllocation()
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
    }

    void connectForwardLayers(size_t layerId)
    {
        using namespace services;
        using namespace layers;

        SharedPtr<forward::LayerIface> forwardLayer = _forwardLayers->get(layerId);
        forwardLayer->allocateResult();
        SharedPtr<forward::Result> forwardResult = forwardLayer->getLayerResult();
        const NextLayers &next = _nextLayers->get(layerId);
        for(size_t j = 0; j < next.size(); j++)
        {
            _forwardLayers->get(next[j])->addInput(forwardResult, j, 0 /* index in input object of next[j] forward layer */);
        }
    }

    template<typename modelFPType>
    DAAL_EXPORT void createWeightsAndBiases();

    bool _storeWeightsInTable;              /*!< Flag. True if weights and biases of all the layers are stored in one numeric table */

    services::SharedPtr<neural_networks::ForwardLayers> _forwardLayers; /*!< List of forward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _nextLayers; /*!< List of edges connecting the layers in the network */
    services::SharedPtr<LearnableParametersIface> _weightsAndBiases;
};
}
using interface1::ModelImpl;
using interface1::LearnableParametersIface;
}
}
}
#endif
