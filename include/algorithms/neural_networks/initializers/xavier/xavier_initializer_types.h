/* file: xavier_initializer_types.h */
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
//  Implementation of Xavier initializer.
//--
*/

#ifndef __XAVIER_INITIALIZER_TYPES_H__
#define __XAVIER_INITIALIZER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/initializers/initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
/**
 * @defgroup initializers_xavier Xavier Initializer
 * \copydoc daal::algorithms::neural_networks::initializers::xavier
 * @ingroup initializers
 * @{
 */
/**
 * \brief Contains classes for neural network weights and biases Xavier initializer
 */
namespace xavier
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__XAVIER__METHOD"></a>
 * Available methods to compute Xavier initializer
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__XAVIER__PARAMETER"></a>
 * \brief Xavier initializer parameters
 */
class DAAL_EXPORT Parameter : public initializers::Parameter
{
public:
    /**
     *  Main constructor
     *  \param[in] _seed Seed for generating random numbers for the initialization \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(size_t _seed = 777): seed(_seed) {}

    size_t seed; /*!< Seed for generating random numbers \DAAL_DEPRECATED_USE{ engine } */

    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace xavier
/** @} */
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
