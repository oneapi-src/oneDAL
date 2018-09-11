/* file: concat_layer_types.h */
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
//  Implementation of the concat layer
//--
*/

#ifndef __CONCAT_LAYER_TYPES_H__
#define __CONCAT_LAYER_TYPES_H__

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
 * @defgroup concat Concat Layer
 * \copydoc daal::algorithms::neural_networks::layers::concat
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the concat layer
 */
namespace concat
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__METHOD"></a>
 * Computation methods for the concat layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward concat layer and results for the forward concat layer
 */

enum LayerDataId
{
    auxInputDimensions = layers::lastLayerInputLayout + 1,  /*!< Numeric table of dimensions along which concatenation is implemented*/
    lastLayerDataId = auxInputDimensions
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__PARAMETER"></a>
 * \brief concat layer parameters
 */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the forward concat layer
    *  \param[in] concatDimension   Index of dimension along which concatenation is implemented
    */
    Parameter(size_t concatDimension = 0);

    size_t concatDimension;    /*!< Index of dimension along which concatenation is implemented*/
};

} // namespace interface1
using interface1::Parameter;

} // namespace concat
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
