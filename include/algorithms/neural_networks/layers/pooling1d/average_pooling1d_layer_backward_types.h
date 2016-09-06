/* file: average_pooling1d_layer_backward_types.h */
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
//  Implementation of backward average 1D pooling layer.
//--
*/

#ifndef __AVERAGE_POOLING1D_LAYER_BACKWARD_TYPES_H__
#define __AVERAGE_POOLING1D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/pooling1d/pooling1d_layer_backward_types.h"
#include "algorithms/neural_networks/layers/pooling1d/average_pooling1d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling1d
{
/**
 * @defgroup average_pooling1d_backward Backward One-dimensional Average Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::average_pooling1d::backward
 * @ingroup average_pooling1d
 * @{
 */
/**
 * \brief Contains classes for backward average 1D pooling layer
 */
namespace backward
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING1D__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward average 1D pooling layer
 */
class Input : public pooling1d::backward::Input
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
     * Returns an input object for backward average 1D pooling layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*inputData)[id]);
    }

    /**
     * Sets an input object for the backward average 1D pooling layer
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the object
     */
    void set(LayerDataId id, const data_management::NumericTablePtr &ptr)
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        (*inputData)[id] = ptr;
    }

    /**
     * Checks an input object for the backward average 1D pooling layer
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        pooling1d::backward::Input::check(parameter, method);
    }

    /**
     * Return the collection with gradient size
     * \return The collection with gradient size
     */
    virtual services::Collection<size_t> getGradientSize() const DAAL_C11_OVERRIDE
    {
        services::Collection<size_t> dims;
        data_management::NumericTablePtr inputDims = get(auxInputDimensions);
        if (!inputDims)
        { this->_errors->add(services::ErrorNullInputNumericTable); return dims; }

        data_management::BlockDescriptor<int> block;
        inputDims->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *inputDimsArray = block.getBlockPtr();
        for(size_t i = 0; i < inputDims->getNumberOfColumns(); i++)
        {
            dims.push_back((size_t) inputDimsArray[i]);
        }
        inputDims->releaseBlockOfRows(block);
        return dims;
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING1D__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward average 1D pooling layer
 */
class Result : public pooling1d::backward::Result
{
public:
    /** Default constructor */
    Result() {}
    virtual ~Result() {}

    /**
     * Returns an result object for the backward average 1D pooling layer
     */
    using layers::backward::Result::get;

    /**
     * Sets an result object for the backward average 1D pooling layer
     */
    using layers::backward::Result::set;

    /**
     * Allocates memory to store the result of the backward average 1D pooling layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the backward average 1D pooling layer
     * \param[in] method Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        pooling1d::backward::Result::allocate<algorithmFPType>(input, parameter, method);
    }

    /**
     * Checks the result of the backward average 1D pooling layer
     * \param[in] input     %Input object for the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        pooling1d::backward::Result::check(input, parameter, method);
    }

    /**
     * Returns the serialization tag of the backward 1D average pooling layer result
     * \return     Serialization tag of the backward 1D average pooling layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING1D_BACKWARD_RESULT_ID; }

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

} // namespace average_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
