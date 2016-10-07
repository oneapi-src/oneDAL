/* file: concat_layer_forward_types.h */
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
//  Implementation of the forward concat layer
//--
*/

#ifndef __CONCAT_LAYER_FORWARD_TYPES_H__
#define __CONCAT_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the concat layer
 */
namespace concat
{
/**
 * @defgroup concat_forward Forward Concat Layer
 * \copydoc daal::algorithms::neural_networks::layers::concat::forward
 * @ingroup concat
 * @{
 */
/**
 * \brief Contains classes for the forward concat layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward concat layer
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    /** \brief Default constructor */
    Input();

    /**
    * Gets the input of the forward concat layer
    */
    using layers::forward::Input::get;

    /**
    * Sets the input of the forward concat layer
    */
    using layers::forward::Input::set;

    virtual ~Input() {}

    /**
    * Returns input Tensor of the forward concat layer
    * \param[in] id       Identifier of the input object
    * \param[in] index    Index of the input object
    * \return             %Input tensor that corresponds to the given identifier
    */
    services::SharedPtr<data_management::Tensor> get(layers::forward::InputLayerDataId id, size_t index) const;

    /**
    * Returns input LayerData of the forward concat layer
    * \param[in] id    Identifier of the input object
    * \return          %Input InputLayerData that corresponds to the given identifier
    */
    services::SharedPtr<LayerData> get(layers::forward::InputLayerDataId id) const;

    /**
    * Sets input for the forward concat layer
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(layers::forward::InputLayerDataId id, const services::SharedPtr<LayerData> &value);

    /**
    * Sets input for the forward concat layer
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    * \param[in] index   Index of the input object
    */
    void set(layers::forward::InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index);

    /**
     * Adds tensor with data to the input object of the forward concat layer
     * \param[in] dataTensor    Tensor with data
     * \param[in] index         Index of the tensor with data
     */
    virtual void addData(const services::SharedPtr<data_management::Tensor> &dataTensor, size_t index) DAAL_C11_OVERRIDE;

    /**
     * Erases input data tensor from the input of the forward layer
     */
    virtual void eraseInputData() DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for the forward concat layer
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns the layout of the input object for the layer algorithm
    * \return Layout of the input object for the layer algorithm
    */
    LayerInputLayout getLayout() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the forward concat layer
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    /** \brief Default constructor */
    Result();
    virtual ~Result() {};

    /**
     * Returns the result of the forward concat layer
     */
    using layers::forward::Result::get;

    /**
    * Sets the result of the forward concat layer
    */
    using layers::forward::Result::set;

    /**
    * Sets the result of the forward concat layer
    * \param[in] id      Identifier of the result
    * \param[in] value   Pointer to the result
    */
    void set(LayerDataId id, const data_management::NumericTablePtr &value);

    /**
    * Returns input object of the forward concat layer
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(layers::concat::LayerDataId id) const;

    /**
    * Returns collection of dimensions of concat layer output
    * \param[in] inputSize   Collection of input tensors dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of concat layer output
    */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns collection of dimensions of concat layer output
    * \param[in] inputSize   Collection of input tensors dimensions
    * \param[in] parameter   Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of concat layer output
    */
    services::Collection<size_t> getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                              const daal::algorithms::Parameter *parameter, const int method) DAAL_C11_OVERRIDE;
    /**
    * Allocates memory to store the result of the forward concat layer
    * \param[in] input     Pointer to an object containing the input data
    * \param[in] parameter %Parameter of the algorithm
    * \param[in] method    Computation method for the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
    * Checks the result object for the layer algorithm
    * \param[in] input         %Input of the algorithm
    * \param[in] parameter     %Parameter of algorithm
    * \param[in] method        Computation method of the algorithm
    */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                       int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns the serialization tag of the result
    * \return     Serialization tag of the result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_FORWARD_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
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
} // namespace forward
/** @} */
} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
