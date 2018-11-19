/* file: lrn_layer_types.h */
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
//  Implementation of the local response normalization layer types.
//--
*/

#ifndef __LRN_LAYER_TYPES_H__
#define __LRN_LAYER_TYPES_H__

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
 * @defgroup lrn Local Response Normalization Layer
 * \copydoc daal::algorithms::neural_networks::layers::lrn
 * @ingroup layers
 * @{
 */
namespace lrn
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__METHOD"></a>
 * \brief Computation methods for the local response normalization layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward stage and results for the forward stage of the local response normalization layer
 */
enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1, /*!< Data processed at the forward stage of the layer */
    auxSmBeta, /*!< Pointer to the tensor of size n1 x n2 x ... x np, that stores value of (kappa + alpha * sum((x_i)^2))^(-beta) */
    lastLayerDataId = auxSmBeta
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__PARAMETER"></a>
 * \brief Parameters for the local response normalization layer
 *
 * \snippet neural_networks/layers/lrn/lrn_layer_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the local response normalization layer
    *  \param[in] dimension_ Numeric table of size 1 x 1 with index of type size_t to calculate local response normalization.
    *  \param[in] kappa_     Value of hyper-parameter kappa
    *  \param[in] alpha_     Value of hyper-parameter alpha
    *  \param[in] beta_      Value of hyper-parameter beta
    *  \param[in] nAdjust_   Value of hyper-parameter n
    */
    Parameter(
        data_management::NumericTablePtr dimension_ = data_management::HomogenNumericTable<size_t>::create(1, 1, data_management::NumericTableIface::doAllocate, 1),
        const double kappa_ = 2,
        const double alpha_ = 1.0e-04,
        const double beta_ = 0.75,
        const size_t nAdjust_ = 5 );

    data_management::NumericTablePtr dimension; /*!< Numeric table of size 1 x 1 with index of type size_t
                                                                       to calculate local response normalization. */
    double kappa;     /*!< Value of hyper-parameter kappa */
    double alpha;     /*!< Value of hyper-parameter alpha */
    double beta;      /*!< Value of hyper-parameter beta */
    size_t nAdjust;   /*!< Value of hyper-parameter n */

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace lrn
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
