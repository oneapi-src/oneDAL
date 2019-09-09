/* file: neural_networks_prediction_input.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
    data,        /*!< Input data set */
    lastTensorInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODELINPUT_ID"></a>
 * \brief Available identifiers of input Model objects in the prediction stage
 * of the neural network algorithm
 */
enum ModelInputId
{
    model = lastTensorInputId + 1,        /*!< Input model trained by the neural network algorithm */
    lastModelInputId = model
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
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input& other);

    virtual ~Input() {};

    /**
     * Returns %input object for the neural networks prediction algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(TensorInputId id) const;

    /**
     * Sets %input object for the neural networks prediction algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the parameter
     */
    void set(TensorInputId id, const data_management::TensorPtr &value);

    /**
     * Returns %input object for the neural networks prediction algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(ModelInputId id) const;

    /**
     * Sets %input object for the neural networks prediction algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the parameter
     */
    void set(ModelInputId id, const ModelPtr &value);

    /**
    * Checks %input object for the neural networks algorithm
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computatiom method
    *
     * \return Status of computations
    */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
} // namespace interface1
using interface1::Input;
/** @} */
}
}
}
} // namespace daal
#endif
