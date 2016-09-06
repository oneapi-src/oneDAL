/* file: initializer_types.h */
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


namespace daal
{
namespace algorithms
{
namespace neural_networks
{
/**
 * \brief Contains classes for neural network weights and biases initializers
 */
namespace layers
{
namespace forward
{
namespace interface1
{
class LayerIface;
}
using interface1::LayerIface;
}
}

namespace initializers
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INPUTID"></a>
 * Available identifiers of input objects for neural network weights and biases initializer
 */
enum InputId
{
    data = 0,       /*!< Input data */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__RESULTID"></a>
 * Available identifiers of results for the neural network weights and biases initializer
 */
enum ResultId
{
    value = 0     /*!< Tensor to store the result */
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
    Parameter(services::SharedPtr<layers::forward::LayerIface> _layer = services::SharedPtr<layers::forward::LayerIface>()): layer(_layer) {}

    virtual ~Parameter() {}

    services::SharedPtr<layers::forward::LayerIface> layer; /*!<Pointer to the layer whose weights and biases are initialized by the initializer */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INPUT"></a>
 * \brief %Input objects for initializer algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    /**
     * Default constructor
     */
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns input tensor of the initializer
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(InputId id) const
    {
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the initializer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks an input object for the initializer
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        if (!data_management::checkTensor(get(data).get(), this->_errors.get(), dataStr())) { return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the neural network weights and biases initializer
 */
class Result : public daal::algorithms::Result
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
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);

        set(value, algInput->get(data));
    }

    /**
     * Returns result of the initializer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the initializer
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        Argument::set(id, ptr);
    }

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

    /**
     * Checks the result object for the initializer
     * \param[in] input         %Input of the algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                       int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Input *algInput = static_cast<const Input *>(input);

        if (!data_management::checkTensor(get(value).get(), this->_errors.get(), valueStr(), &(algInput->get(data)->getDimensions()))) { return; }
    }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // interface1
using interface1::Input;
using interface1::Result;
using interface1::Parameter;
} // namespace initializers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
