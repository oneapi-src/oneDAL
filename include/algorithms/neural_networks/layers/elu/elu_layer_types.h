/* file: elu_layer_types.h */
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
//  Implementation of the Exponential Linear Unit (ELU) layer
//--
*/

#ifndef __ELU_LAYER_TYPES_H__
#define __ELU_LAYER_TYPES_H__

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
 * @defgroup elu_layers Exponential Linear Unit (ELU) Layer
 * \copydoc daal::algorithms::neural_networks::layers::elu
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the ELU layer
 */
namespace elu
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__METHOD"></a>
 * Computation methods for the ELU layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward ELU layer and results for the forward ELU layer
 */
enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1, /*!< Data processed at the forward stage of the layer */
    auxIntermediateValue,
    lastLayerDataId = auxIntermediateValue
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__PARAMETER"></a>
 * \brief Parameters for the ELU layer
 *
 * \snippet neural_networks/layers/elu/elu_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the ELU layer
     */
    Parameter(double alpha = 1.0);

    double alpha; /*!< Coefficient for the ELU layer */
};
/* [Parameter source code] */
} // namespace interface1
using interface1::Parameter;

} // namespace elu
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
