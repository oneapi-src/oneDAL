/* file: uniform_initializer_types.h */
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
//  Implementation of uniform initializer.
//--
*/

#ifndef __UNIFORM_INITIALIZER_TYPES_H__
#define __UNIFORM_INITIALIZER_TYPES_H__

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
 * @defgroup initializers_uniform Uniform Initializer
 * \copydoc daal::algorithms::neural_networks::initializers::uniform
 * @ingroup initializers
 * @{
 */
/**
 * \brief Contains classes for neural network weights and biases uniform initializer
 */
namespace uniform
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__METHOD"></a>
 * Available methods to compute uniform initializer
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__PARAMETER"></a>
 * \brief Uniform initializer parameters
 */
class Parameter: public initializers::Parameter
{
public:
    /**
     *  Main constructor
     *  \param[in] _a    Left bound a
     *  \param[in] _b    Right bound b
     *  \param[in] _seed Seed for generating random numbers for the initialization \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(double _a = -0.5, double _b = 0.5, size_t _seed = 777): a(_a), b(_b), seed(_seed) {}

    double a;    /*!< Left bound a */
    double b;    /*!< Right bound b */
    size_t seed; /*!< Seed for generating random numbers \DAAL_DEPRECATED_USE{ engine } */
};

} // namespace interface1
using interface1::Parameter;

} // namespace uniform
/** @} */
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
