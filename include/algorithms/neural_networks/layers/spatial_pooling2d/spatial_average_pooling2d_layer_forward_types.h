/* file: spatial_average_pooling2d_layer_forward_types.h */
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
//  Implementation of forward spatial pyramid average 2D pooling layer.
//--
*/

#ifndef __SPATIAL_AVERAGE_POOLING2D_LAYER_FORWARD_TYPES_H__
#define __SPATIAL_AVERAGE_POOLING2D_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_forward_types.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_average_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_average_pooling2d
{
/**
 * @defgroup spatial_average_pooling2d_forward Forward Two-dimensional Spatial pyramid average Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::spatial_average_pooling2d::forward
 * @ingroup spatial_average_pooling2d
 * @{
 */
/**
 * \brief Contains classes for forward spatial pyramid average 2D pooling layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward spatial pyramid average 2D pooling layer
 * See \ref pooling2d::forward::interface1::Input "pooling2d::forward::Input"
 */
class DAAL_EXPORT Input : public spatial_pooling2d::forward::Input
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward spatial pyramid average 2D pooling layer
 */
class DAAL_EXPORT Result : public spatial_pooling2d::forward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /** Default constructor */
    Result();
    virtual ~Result() {}

    using layers::forward::Result::get;
    using layers::forward::Result::set;

    /**
     * Allocates memory to store the result of the forward spatial pyramid average 2D pooling layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the forward spatial pyramid average 2D pooling layer
     * \param[in] method Computation method for the layer
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns the result of the forward spatial pyramid average 2D pooling layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(LayerDataId id) const;

    /**
     * Sets the result of the forward spatial pyramid average 2D pooling layer
     * \param[in] id Identifier of the result
     * \param[in] ptr Result
     */
    void set(LayerDataId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result of the forward spatial pyramid average 2D pooling layer
     * \param[in] input     %Input of the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace forward
/** @} */
} // namespace spatial_average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
