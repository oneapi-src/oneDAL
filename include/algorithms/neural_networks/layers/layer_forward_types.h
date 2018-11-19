/* file: layer_forward_types.h */
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
//  Implementation of neural_networks Network layer.
//--
*/

#ifndef __LAYERS__FORWARD__TYPES__H__
#define __LAYERS__FORWARD__TYPES__H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
/**
 * @defgroup layers_forward Forward Base Layer
 * \copydoc daal::algorithms::neural_networks::layers::forward
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the forward stage of the neural network layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUTID"></a>
 * Available identifiers of input objects for the layer algorithm
 */
enum InputId
{
    data,       /*!< Input data */
    weights,    /* Weights of the neural network layer */
    biases,       /* Biases of the neural network layer */
    lastInputId = biases
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUTLAYERDATAID"></a>
 * Available identifiers of input objects for the layer algorithm
 */
enum InputLayerDataId
{
    inputLayerData = lastInputId + 1,
    lastInputLayerDataId = inputLayerData
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__RESULTID"></a>
 * Available identifiers of results for the layer algorithm
 */
enum ResultId
{
    value,      /*!< Table to store the result */
    lastResultId = value
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__RESULTLAYERDATAID"></a>
 * Available identifiers of results for the layer algorithm
 */
enum ResultLayerDataId
{
    resultForBackward = lastResultId + 1,  /*!< Data for backward step */
    lastResultLayerDataId = resultForBackward
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUTIFACE"></a>
 * \brief Abstract class that specifies interface of the input objects for the neural network layer algorithm
 */
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    /** \brief Constructor
    * \param[in] nElements    Number of elements in Input structure
    */
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    InputIface(const InputIface& other);
    virtual ~InputIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUT"></a>
 * \brief %Input objects for layer algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    /**
     * Constructs input objects for the forward layer of neural network
     * \param[in] nElements     Number of input objects for the forward layer
     */
    Input(size_t nElements = lastInputLayerDataId + 1);
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns input Tensor of the layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    data_management::TensorPtr get(forward::InputId id) const;

    /**
     * Sets input for the layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::TensorPtr &ptr);

    /**
    * Returns input InputLayerData of the layer algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input InputLayerData that corresponds to the given identifier
    */
    LayerDataPtr get(forward::InputLayerDataId id) const;

    /**
     * Sets input for the layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputLayerDataId id, const LayerDataPtr &ptr);

    /**
     * Checks an input object for the layer algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the layout of the input object for the layer algorithm
     * \return Layout of the input object for the layer algorithm
     */
    virtual LayerInputLayout getLayout();

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const;

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const;

    /**
     * Adds tensor with data to the input object of the layer algorithm
     * \param[in] dataTensor    Tensor with data
     * \param[in] index         Index of the tensor with data
     *
     * \return Status of computations
     */
    virtual services::Status addData(const data_management::TensorPtr &dataTensor, size_t index);

    /**
     * Erases input data tensor from the input of the forward layer
     *
     * \return Status of computations
     */
    virtual services::Status eraseInputData()
    {
        set(layers::forward::data, data_management::TensorPtr());
        return services::Status();
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the layer algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
    DECLARE_SERIALIZABLE_IMPL();
public:
    DAAL_CAST_OPERATOR(Result);
    /** \brief Constructor */
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of layer
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        return services::Status();
    }

    /**
    * Returns collection of dimensions of layer output
    * \param[in] inputSize   Collection of input tensor dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of layer output
    */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const = 0;

    /**
    * Returns collection of dimensions of layer output
    * \param[in] inputSize Collection of input tensors dimensions
    * \param[in] par       Parameters of the algorithm
    * \param[in] method    Method of the algorithm
    * \return    Collection of dimensions of layer output
    */
    virtual services::Collection<size_t> getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                                      const daal::algorithms::Parameter *par, const int method);

    /**
    * Returns collection of dimensions of layer output
    * \param[in] inputSize   Collection of input tensor dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of layer output
    */
    virtual services::Collection< services::Collection<size_t> > getValueCollectionSize(const services::Collection<size_t> &inputSize,
                                                                                        const daal::algorithms::Parameter *par, const int method);

    /**
     * Returns result of the layer algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultId id) const;

    /**
     * Returns result of the layer algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    LayerDataPtr get(ResultLayerDataId id) const;

    /**
     * Sets the result of the layer algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::TensorPtr &ptr);

    /**
    * Sets the result of the layer algorithm
    * \param[in] id    Identifier of the result
    * \param[in] ptr   Pointer to the result
    */
    void set(ResultLayerDataId id, const LayerDataPtr &ptr);

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_FORWARD_RESULT_ID; }

    /**
     * Checks the result object for the layer algorithm
     * \param[in] input         %Input of the algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     *
     * \return Status of computations
     */
    virtual services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                                   int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the layout of the result object for the layer algorithm
     * \return Layout of the result object for the layer algorithm
     */
    virtual LayerResultLayout getLayout();

    /**
     * Sets the result that is used in backward layer
     * \param[in] input     Pointer to an object containing the input data
     *
     * \return Status of computations
     */
    virtual services::Status setResultForBackward(const daal::algorithms::Input *input)
    {
        return services::Status();
    }

    /**
     * Returns resulting value of the layer algorithm
     * \param[in] index Index of the tensor with value
     * \return Resulting value that corresponds to the given index
     */
    virtual data_management::TensorPtr getValue(size_t index) const;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // interface1
using interface1::InputIface;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
} // forward
/** @} */
} // namespace layer
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
