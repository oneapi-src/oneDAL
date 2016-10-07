/* file: batch_normalization_layer_forward_types.h */
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
//  Implementation of the forward batch normalization layer.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_TYPES_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
/**
 * @defgroup batch_normalization_forward Forward Batch Normalization Layer
 * \copydoc daal::algorithms::neural_networks::layers::batch_normalization::forward
 * @ingroup batch_normalization
 * @{
 */
/**
 * \brief Contains classes for forward batch normalization layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__INPUTLAYERDATAID"></a>
 * Available identifiers of input objects for the forward batch normalization layer
 */
enum InputLayerDataId
{
    populationMean     = 3, /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population mean computed on the previous stage */
    populationVariance = 4  /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population variance computed on the previous stage */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward batch normalization layer.
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    /** Default constructor */
    Input();

    virtual ~Input() {}

    /**
     * Returns an input object for the forward batch normalization layer
     */
    using layers::forward::Input::get;
    /**
    * Sets an input object for the forward batch normalization layer
    */
    using layers::forward::Input::set;


    /**
     * Allocates memory to store input objects of forward batch normalization layer
     * \param[in] parameter %Parameter of forward batch normalization layer
     * \param[in] method    Computation method for the layer
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE;

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE;

    /**
     * Returns an input object for forward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(InputLayerDataId id) const;

    /**
     * Sets input for the forward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object to set
     */
    void set(InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr);

    /**
     * Checks input object of the forward batch normalization layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward batch normalization layer
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    /** Default Constructor */
    Result();
    virtual ~Result() {}

    using layers::forward::Result::get;
    using layers::forward::Result::set;

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory to store the result of the forward batch normalization layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the forward batch normalization layer
     * \param[in] method Computation method for the layer
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Sets the result that is used in backward batch normalization layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE;

    /**
     * Returns the result of the forward batch normalization layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const;

    /**
     * Sets the result of the forward batch normalization layer
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Result
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr);

    /**
     * Checks the result of the forward batch normalization layer
     * \param[in] input     %Input of the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the serialization tag of the forward batch normalization layer result
     * \return     Serialization tag of the forward batch normalization layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_BATCH_NORMALIZATION_FORWARD_RESULT_ID; }

    /**
     * Serializes the object
     * \param[in]  arch  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
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

} // namespace forward

/** @} */
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
