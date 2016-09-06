/* file: stochastic_pooling2d_layer_backward_types.h */
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
//  Implementation of backward stochastic 2D pooling layer.
//--
*/

#ifndef __STOCHASTIC_POOLING2D_LAYER_BACKWARD_TYPES_H__
#define __STOCHASTIC_POOLING2D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/pooling2d/pooling2d_layer_backward_types.h"
#include "algorithms/neural_networks/layers/pooling2d/stochastic_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace stochastic_pooling2d
{
/**
 * @defgroup stochastic_pooling2d_backward Backward Two-dimensional Stochastic Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::stochastic_pooling2d::backward
 * @ingroup stochastic_pooling2d
 * @{
 */
/**
 * \brief Contains classes for backward stochastic 2D pooling layer
 */
namespace backward
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__STOCHASTIC_POOLING2D__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward stochastic 2D pooling layer
 */
class Input : public pooling2d::backward::Input
{
public:
    /**
     * Default constructor
     */
    Input() {}

    virtual ~Input() {}

    using layers::backward::Input::get;
    using layers::backward::Input::set;

    /**
     * Returns an input object for backward stochastic 2D pooling layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*inputData)[id]);
    }

    /**
     * Returns an input object for backward stochastic 2D pooling layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(LayerDataNumericTableId id) const
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*inputData)[id]);
    }

    /**
     * Sets an input object for the backward stochastic 2D pooling layer
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the object
     */
    void set(LayerDataId id, const data_management::TensorPtr &ptr)
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        (*inputData)[id] = ptr;
    }

    /**
     * Sets an input object for the backward stochastic 2D pooling layer
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the object
     */
    void set(LayerDataNumericTableId id, const data_management::NumericTablePtr &ptr)
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        (*inputData)[id] = ptr;
    }

    /**
     * Checks an input object for the backward stochastic 2D pooling layer
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        pooling2d::backward::Input::check(parameter, method);
    }

protected:
    virtual data_management::NumericTablePtr getAuxInputDimensions() const DAAL_C11_OVERRIDE
    {
        return get(auxInputDimensions);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__STOCHASTIC_POOLING2D__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward stochastic 2D pooling layer
 */
class Result : public pooling2d::backward::Result
{
public:
    /**
     * Default constructor
     */
    Result() {}
    virtual ~Result() {}

    using layers::backward::Result::get;
    using layers::backward::Result::set;

    /**
     * Allocates memory to store the result of the backward stochastic 2D pooling layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the backward stochastic 2D pooling layer
     * \param[in] method Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        pooling2d::backward::Result::allocate<algorithmFPType>(input, parameter, method);
    }

    /**
     * Checks the result of the backward stochastic 2D pooling layer
     * \param[in] input     %Input object for the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        pooling2d::backward::Result::check(input, parameter, method);
    }

    /**
     * Returns the serialization tag of the backward stochastic 2D pooling layer result
     * \return     Serialization tag of the backward stochastic 2D pooling layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_STOCHASTIC_POOLING2D_BACKWARD_RESULT_ID; }

    /**
     * Serializes the object
     * \param[in]  arch  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes the object
     * \param[in]  arch  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};

} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward
/** @} */
} // namespace stochastic_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
