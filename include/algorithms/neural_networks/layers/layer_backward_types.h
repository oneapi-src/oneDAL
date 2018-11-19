/* file: layer_backward_types.h */
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

#ifndef __LAYERS__BACKWARD__TYPES__H__
#define __LAYERS__BACKWARD__TYPES__H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

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
 * @defgroup layers_backward Backward Base Layer
 * \copydoc daal::algorithms::neural_networks::layers::backward
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the backward stage of the neural network layer
 */
namespace backward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__INPUTID"></a>
 * Available identifiers of input objects for the layer algorithm
 */
enum InputId
{
    inputGradient,   /*!< Gradient of the preceding layer */
    lastInputId = inputGradient
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__INPUTLAYERDATAID"></a>
 * Available identifiers of input objects for the layer algorithm
 */
enum InputLayerDataId
{
    inputFromForward = lastInputId + 1,   /*!< Input structure retrieved from the result of the forward layer */
    lastInputLayerDataId = inputFromForward
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__RESULTID"></a>
 * Available identifiers of results for the layer algorithm
 */
enum ResultId
{
    gradient,     /*!< Gradient with respect to the outputs */
    weightDerivatives, /*!< Gradient with respect to the weight */
    biasDerivatives, /*!< Gradient with respect to the bias */
    lastResultId = biasDerivatives
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__RESULTLAYERDATAID"></a>
 * Available identifiers of result objects for the layer algorithm
 */
enum ResultLayerDataId
{
    resultLayerData = lastResultId + 1,
    lastResultLayerDataId = resultLayerData
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__INPUTIFACE"></a>
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__INPUT"></a>
 * \brief %Input parameters for the layer algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    /** \brief Constructor */
    Input();
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns input Tensor of the layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputId id) const;

    /**
     * Returns input Tensor of the layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    LayerDataPtr get(InputLayerDataId id) const;

    /**
     * Sets input for the layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::TensorPtr &ptr);

    /**
     * Sets input for the layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputLayerDataId id, const LayerDataPtr &ptr);

    /**
     * Adds tensor with input gradient to the input object of the layer algorithm
     * \param[in] igTensor  Tensor with input gradient
     * \param[in] index     Index of the tensor with input gradient
     *
     * \return Status of computations
     */
    virtual services::Status addInputGradient(const data_management::TensorPtr &igTensor, size_t index);

    /**
     * Sets input structure retrieved from the result of the forward layer
     * \param[in] result Result of the forward layer
     *
     * \return Status of computations
     */
    virtual services::Status setInputFromForward(forward::ResultPtr result);

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
    virtual LayerInputLayout getLayout() const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__RESULT"></a>
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
     * Returns result of the layer algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultId id) const;

    /**
     * Sets the result of the layer algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::TensorPtr &ptr);

    /**
    * Returns result InputLayerData of the layer algorithm
    * \param[in] id    Identifier of the result object
    * \return          Resulting InputLayerData that corresponds to the given identifier
    */
    LayerDataPtr get(backward::ResultLayerDataId id) const;

    /**
     * Sets result for the layer algorithm
     * \param[in] id    Identifier of the result object
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultLayerDataId id, const LayerDataPtr &ptr);

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_BACKWARD_RESULT_ID; }

    /**
     * Checks the result object for the layer algorithm
     * \param[in] input         %Input of algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                           int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the layout of the result object for the layer algorithm
     * \return Layout of the result object for the layer algorithm
     */
    virtual LayerResultLayout getLayout() const;

    /**
     * Returns resulting gradient of the layer algorithm
     * \param[in] index Index of the tensor with gradient
     * \return Resulting gradient that corresponds to the given index
     */
    virtual data_management::TensorPtr getGradient(size_t index) const;
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
using interface1::InputIface;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
} // backward

/** @} */
} // namespace layer
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
