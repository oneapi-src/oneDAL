/* file: fullyconnected_layer_types.h */
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
//  Fully-connected layer parameter structure.
//--
*/

#ifndef __FULLYCONNECTED_LAYER_TYPES_H__
#define __FULLYCONNECTED_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup fullyconnected Fully-connected Layer
 * \copydoc daal::algorithms::neural_networks::layers::fullyconnected
 * @ingroup layers
 * @{
 */
namespace fullyconnected
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__METHOD"></a>
 * Available methods to compute forward and backward fully-connected layer
 */
enum Method
{
    defaultDense = 0,    /*!< Default: performance-oriented method. */
};

/**
 * Available identifiers of results of the forward fully-connected layer
 * and input objects for the backward fully-connected layer
 */
enum LayerDataId
{
    auxData    = 0, /*!< Data processed at the forward stage of the layer */
    auxWeights = 1, /*!< Weights used at the forward stage of the layer */
    lastLayerDataId = auxWeights
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__PARAMETER"></a>
 * \brief Fully-connected layer parameters
 */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
     *  Main constructor
     *  \param[in] _nOutputs A number of layer outputs m. The parameter required to initialize the layer
     */
    Parameter(size_t _nOutputs);

    size_t nOutputs; /*!< A number of layer outputs. The parameter required to initialize the layer */
};

} // namespace interface1
using interface1::Parameter;

} // namespace fullyconnected
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
