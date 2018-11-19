/* file: initializer.h */
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
//  Implementation of neural network layer.
//--
*/

#ifndef __INITIALIZERS_H__
#define __INITIALIZERS_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
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
 * @ingroup initializers
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INITIALIZERCONTAINERIFACE"></a>
 * \brief Class that specifies interfaces of implementations of the neural network weights and biases initializer
 */
class InitializerContainerIface : public AnalysisContainerIface<batch>
{
public:
    virtual ~InitializerContainerIface()
    {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INITIALIZERIFACE"></a>
 *  \brief Class representing a neural network weights and biases initializer
 */
class InitializerIface : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::neural_networks::initializers::Input     InputType;
    typedef algorithms::neural_networks::initializers::Parameter ParameterType;
    typedef algorithms::neural_networks::initializers::Result    ResultType;

    InputType  input;   /*!< Input of the initializer */

    InitializerIface() {}
    InitializerIface(const InitializerIface& other) {}

    virtual ~InitializerIface() {}

    /**
     * Get parameters of the initializer
     * \return Parameters of the initializer
     */
    virtual ParameterType * getParameter() = 0;
};
typedef services::SharedPtr<InitializerIface> InitializerIfacePtr;

} // namespace interface1
using interface1::InitializerContainerIface;
using interface1::InitializerIface;
using interface1::InitializerIfacePtr;
/** @} */
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
