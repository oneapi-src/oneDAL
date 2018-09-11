/* file: split_layer_types.h */
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
//  Implementation of the split layer
//--
*/

#ifndef __SPLIT_LAYER_TYPES_H__
#define __SPLIT_LAYER_TYPES_H__

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
 * @defgroup split Split Layer
 * \copydoc daal::algorithms::neural_networks::layers::split
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the split layer
 */
namespace split
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__METHOD"></a>
 * Computation methods for the split layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__PARAMETER"></a>
 * \brief split layer parameters
 */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the forward split layer
    *  \param[in] nOutputs   Number of outputs for forward split layer
    *  \param[in] nInputs    Number of inputs for backward split layer
    */
    Parameter(size_t nOutputs = 1, size_t nInputs = 1);

    size_t nOutputs;    /*!< Number of outputs for forward split layer*/
    size_t nInputs;    /*!< Number of inputs for backward split layer*/
};

} // namespace interface1
using interface1::Parameter;

} // namespace split
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
