/* file: initializer_types.h */
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

#ifndef __INITIALIZERS__TYPES__H__
#define __INITIALIZERS__TYPES__H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "algorithms/neural_networks/initializers/initializer_types_defs.h"
#include "algorithms/engines/engine.h"
#include "algorithms/engines/mt19937/mt19937.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
/**
 * @defgroup initializers Initializers
 * \copydoc daal::algorithms::neural_networks::initializers
 * @ingroup neural_networks
 * @{
 */
/**
 * \brief Contains classes for neural network weights and biases initializers
 */
namespace initializers
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INPUTID"></a>
 * Available identifiers of input objects for neural network weights and biases initializer
 */
enum InputId
{
    data,      /*!< Input data */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__RESULTID"></a>
 * Available identifiers of results for the neural network weights and biases initializer
 */
enum ResultId
{
    value,      /*!< Tensor to store the result */
    lastResultId = value
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__PARAMETER"></a>
 * Parameters of the neural network weights and biases initializer
 */
class Parameter: public daal::algorithms::Parameter
{
public:
    Parameter(layers::forward::LayerIfacePtr layerForParameter = layers::forward::LayerIfacePtr()): layer(layerForParameter) {}
    virtual ~Parameter() {}

    layers::forward::LayerIfacePtr layer; /*!< Pointer to the layer whose weights and biases are initialized by the initializer */
    engines::EnginePtr engine;            /*!< Pointer to the engine for generating random numbers */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INPUT"></a>
 * \brief %Input objects for initializer algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /**
     * Default constructor
     */
    Input() : daal::algorithms::Input(1) {}
    /** Copy constructor */
    Input(const Input& other) : daal::algorithms::Input(other){}

    virtual ~Input() {}

    /**
     * Returns input tensor of the initializer
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputId id) const
    {
        return data_management::Tensor::cast(Argument::get(id));
    }

    /**
     * Sets input for the initializer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::TensorPtr &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks an input object for the initializer
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the neural network weights and biases initializer
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** \brief Constructor */
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {}

    /**
     * Allocates memory to store the results of initializer
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);

        set(value, algInput->get(data));
        return services::Status();
    }

    /**
     * Returns result of the initializer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultId id) const
    {
        return data_management::Tensor::cast(Argument::get(id));
    }

    /**
     * Sets the result of the initializer
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::TensorPtr &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result object for the initializer
     * \param[in] input         %Input of the algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     *
     * \return Status of computations
     */
    virtual services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
        int method) const DAAL_C11_OVERRIDE;

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
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::Parameter;
} // namespace initializers
/** @} */
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
