/* file: neural_networks_training_types.h */
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

#ifndef __NEURAL_NETWORKS_TRAINING_TYPES_H__
#define __NEURAL_NETWORKS_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "neural_networks_training_input.h"
#include "neural_networks_training_result.h"
#include "neural_networks_training_partial_result.h"
#include "algorithms/neural_networks/layers/layer.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * @defgroup neural_networks_training Training
 * \copydoc daal::algorithms::neural_networks::training
 * @ingroup neural_networks
 * @{
 */
/**
* \brief Contains classes for training the model of the neural network
*/
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__METHOD"></a>
 * \brief Computation methods for the neural network model based training
 */
enum Method
{
    defaultDense = 0,     /*!< Default: Feedforward neural network */
    feedforwardDense = 0  /*!< Feedforward neural network */
};

}
/** @} */
}
}
} // namespace daal
#endif
