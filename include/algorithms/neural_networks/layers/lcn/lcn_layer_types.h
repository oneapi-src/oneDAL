/* file: lcn_layer_types.h */
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
//  Implementation of the local contrast normalization layer.
//--
*/

#ifndef __NEURAL_NETWORKS__LCN_LAYER_TYPES_H__
#define __NEURAL_NETWORKS__LCN_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
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
 * @defgroup lcn_layers Local contrast normalization (LCN) Layer
 * \copydoc daal::algorithms::neural_networks::layers::lcn
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes of the lcn layer
 */
namespace lcn
{
/**
 * Available methods to compute forward and backward local contrast normalization layer
 */
enum Method
{
    defaultDense = 0,    /*!< Default: performance-oriented method. */
};

/**
 * Available identifiers of results of the forward local contrast normalization layer
 * and input objects for the backward local contrast normalization layer
 */
enum LayerDataId
{
    auxCenteredData,
    auxSigma,
    auxC,
    auxInvMax,
    lastLayerDataId = auxInvMax
};

/**
 * \brief Data structure representing the indices of the two dimensions on which local contrast normalization is performed
 */
struct Indices
{
    /**
    * Constructs the structure representing the indices of the two dimensions on which local contrast normalization is performed
    * \param[in]  first  The first dimension index
    * \param[in]  second The second dimension index
    */
    Indices(size_t first, size_t second) { dims[0] = first; dims[1] = second; }
    size_t dims[2];
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__PARAMETER"></a>
 * \brief local contrast normalization layer parameters
 */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
     *  Default constructor
     */
    Parameter();

    data_management::TensorPtr kernel;  /*!< Tensor with sizes m_1 x m_2 of the two-dimensional kernel */
    data_management::NumericTablePtr sumDimension;  /*!< Numeric table of size 1x1 that stores dimension f */
    Indices indices; /*!< Data structure representing the dimension for local contrast normalization kernels. (2,3) is supported now */
    double sigmaDegenerateCasesThreshold; /*!< The threshold to avoid degenerate cases when calculating 1/auxSigma */

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace lcn
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
