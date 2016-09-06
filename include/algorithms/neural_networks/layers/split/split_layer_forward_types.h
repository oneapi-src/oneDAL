/* file: split_layer_forward_types.h */
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
//  Implementation of the forward split layer
//--
*/

#ifndef __SPLIT_LAYER_FORWARD_TYPES_H__
#define __SPLIT_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/split/split_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the split layer
 */
namespace split
{
/**
 * @defgroup split_forward Forward Split Layer
 * \copydoc daal::algorithms::neural_networks::layers::split::forward
 * @ingroup split
 * @{
 */
/**
 * \brief Contains classes for the forward split layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__FORWARD__RESULTLAYERDATAID"></a>
 * Available identifiers of results for the forward split layer
 */
enum ResultLayerDataId
{
    valueCollection = 1     /*!< Data for backward step */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward split layer
 */
class Input : public layers::forward::Input
{
public:
    /** \brief Default constructor */
    Input() {};

    /**
    * Gets the input of the forward split layer
    */
    using layers::forward::Input::get;

    /**
    * Sets the input of the forward split layer
    */
    using layers::forward::Input::set;

    virtual ~Input() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the forward split layer
 */
class Result : public layers::forward::Result
{
public:
    /** \brief Default constructor */
    Result() : layers::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the forward split layer
     */
    using layers::forward::Result::get;

    /**
    * Sets the result of the forward split layer
    */
    using layers::forward::Result::set;

    /**
     * Returns a tensor with a given index from the result
     * \param[in] id    Identifier of the collection of input tensors
     * \param[in] index Index of the tensor to be returned
     * \return          Pointer to the table with the input tensor
     */
    services::SharedPtr<data_management::Tensor> get(ResultLayerDataId id, size_t index) const
    {
        services::SharedPtr<LayerData> resCollection = get(id);
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*resCollection)[index]);
    }

    /**
     * Returns result of the layer algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<LayerData> get(ResultLayerDataId id) const
    {
        return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the forward split layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultLayerDataId id, const services::SharedPtr<LayerData> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Sets the result of the forward split layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     * \param[in] index   Index of the result
     */
    void set(ResultLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
    {
        services::SharedPtr<LayerData> layerData = this->get(id);
        (*layerData)[index] = value;
    }

    /**
     * Returns the layout of the result object for the layer algorithm
     * \return Layout of the result object for the layer algorithm
     */
    LayerResultLayout getLayout() DAAL_C11_OVERRIDE { return collectionResult; }

    /**
     * Returns resulting value of the forward split layer
     * \param[in] index Index of the tensor with value
     * \return Resulting value that corresponds to the given index
     */
    virtual services::SharedPtr<data_management::Tensor> getValue(size_t index) const DAAL_C11_OVERRIDE
    {
        return get(valueCollection, index);
    }

    /**
     * Checks the result of the forward split layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *parameter = static_cast<const Parameter *>(par);
        size_t nOutputs = parameter->nOutputs;

        services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(layers::forward::data);
        if (!data_management::checkTensor(inputTensor.get(), this->_errors.get(), dataStr())) { return; }

        services::Collection<size_t> inputDims = inputTensor->getDimensions();

        for (size_t i = 0; i < nOutputs; i++)
        {
            if (!data_management::checkTensor(get(valueCollection, i).get(), this->_errors.get(), valueCollectionStr(), &inputDims)) { return; }
        }
    }

    /**
    * Returns collection of dimensions of split layer output
    * \param[in] inputSize   Collection of input tensor dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of split layer output
    */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
    * Returns collection of dimensions of split layer output
    * \param[in] inputSize   Collection of input tensor dimensions
    * \param[in] parameter   Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of split layer output
    */
    services::Collection< services::Collection<size_t> > getValueCollectionSize(const services::Collection<size_t> &inputSize,
                                                                                const daal::algorithms::Parameter *parameter, const int method) DAAL_C11_OVERRIDE
    {
        const Parameter *par = static_cast<const Parameter *>(parameter);
        size_t nOutputs = par->nOutputs;

        services::Collection<services::Collection<size_t> > dimsCollection;

        for (size_t i = 0; i < nOutputs; i++)
        {
            dimsCollection.push_back(inputSize);
        }

        return dimsCollection;
    }

    /**
    * Allocates memory to store the result of the forward split layer
    * \param[in] input        Pointer to an object containing the input data
    * \param[in] parameter    %Parameter of the algorithm
    * \param[in] method       Computation method for the algorithm
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        if (!get(layers::forward::resultForBackward))
        {
            const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
            const Parameter *par = static_cast<const Parameter *>(parameter);

            size_t nOutputs = par->nOutputs;

            services::SharedPtr<LayerData> resultCollection = services::SharedPtr<LayerData>(new LayerData());

            services::SharedPtr<data_management::Tensor> dataTensor = in->get(layers::forward::data);
            const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
            for(size_t i = 0; i < nOutputs; i++)
            {
                (*resultCollection)[i] = services::SharedPtr<data_management::Tensor>(new data_management::HomogenTensor<algorithmFPType>(
                                                                                          dataDims, data_management::Tensor::doAllocate));
            }
            set(layers::forward::resultForBackward, resultCollection);
        }
    }

    /**
    * Returns the serialization tag of the result
    * \return     Serialization tag of the result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_FORWARD_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE
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
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
