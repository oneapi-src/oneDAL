/* file: lcn_layer_types.h */
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
    auxCenteredData = 0,
    auxSigma        = 1,
    auxC            = 2,
    auxInvMax       = 3
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

    services::SharedPtr<data_management::Tensor> kernel;  /*!< Tensor with sizes m_1 x m_2 of the two-dimensional kernel */
    data_management::NumericTablePtr sumDimension;  /*!< Numeric table of size 1x1 that stores dimension f */
    Indices indices; /*!< Data structure representing the dimension for local contrast normalization kernels. (2,3) is supported now */
    double sigmaDegenerateCasesThreshold; /*!< The threshold to avoid degenerate cases when calculating 1/auxSigma */

    /**
     * Checks the correctness of the parameter
     */
    void check() const DAAL_C11_OVERRIDE;
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
