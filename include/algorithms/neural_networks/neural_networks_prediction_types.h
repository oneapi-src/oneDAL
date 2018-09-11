/* file: neural_networks_prediction_types.h */
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

#ifndef __NEURAL_NETWORKS_PREDICTION_TYPES_H__
#define __NEURAL_NETWORKS_PREDICTION_TYPES_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward.h"
#include "algorithms/neural_networks/neural_networks_prediction_input.h"
#include "algorithms/neural_networks/neural_networks_prediction_result.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup neural_networks_prediction Prediction
 * \copydoc daal::algorithms::neural_networks::prediction
 * @ingroup neural_networks
 * @{
 */
/**
 * \brief Contains classes for prediction and prediction using neural network
 */
namespace neural_networks
{
/**
* \brief Contains classes for making prediction based on the trained model
*/
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__METHOD"></a>
 * \brief Computation methods for the neural network model based prediction
 */
enum Method
{
    defaultDense = 0,     /*!< Feedforward neural network */
    feedforwardDense = 0  /*!< Feedforward neural network */
};

}
}
/** @} */
}
} // namespace daal

#endif
