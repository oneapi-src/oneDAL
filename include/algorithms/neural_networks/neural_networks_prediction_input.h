/* file: neural_networks_prediction_input.h */
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
