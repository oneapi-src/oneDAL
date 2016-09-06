/* file: neural_networks_prediction_input.h */
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
//  Implementation of neural network algorithm interface.
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_INPUT_H__
#define __NEURAL_NETWORKS_PREDICTION_INPUT_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_prediction_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for prediction and prediction using neural network
 */
namespace neural_networks
{
namespace prediction
{
/**
 * @ingroup neural_networks_prediction
 * @{
 */
/**
* <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__TENSORINPUTID"></a>
* Available identifiers of input Tensor objects in the prediction stage
* of the neural network algorithm
*/
enum TensorInputId
{
    data = 0        /*!< Input data set */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODELINPUT_ID"></a>
 * \brief Available identifiers of input Model objects in the prediction stage
 * of the neural network algorithm
 */
enum ModelInputId
{
    model = 1,        /*!< Input model trained by the neural network algorithm */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__INPUT"></a>
 * \brief Input objects of the neural networks prediction algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(2) {};

    virtual ~Input() {};

    /**
     * Returns %input object for the neural networks prediction algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(TensorInputId id) const
    {
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets %input object for the neural networks prediction algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the parameter
     */
    void set(TensorInputId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Returns %input object for the neural networks prediction algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<Model> get(ModelInputId id) const
    {
        return services::staticPointerCast<Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets %input object for the neural networks prediction algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the parameter
     */
    void set(ModelInputId id, const services::SharedPtr<Model> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks %input object for the neural networks algorithm
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computatiom method
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        if(!get(data))  { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(!get(model)) { this->_errors->add(services::ErrorNullModel); return; }
    }
};
} // namespace interface1
using interface1::Input;
/** @} */
}
}
}
} // namespace daal
#endif
