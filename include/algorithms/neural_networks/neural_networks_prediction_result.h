/* file: neural_networks_prediction_result.h */
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
//  Implementation of neural network algorithm interface.
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_RESULT_H__
#define __NEURAL_NETWORKS_PREDICTION_RESULT_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "neural_networks_prediction_input.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for prediction and prediction using neural network
 */
namespace neural_networks
{
namespace prediction
{
/**
 * @ingroup neural_networks_prediction
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__RESULTID"></a>
 * Available identifiers of results obtained in the prediction stage of the neural network algorithm
 */
enum ResultId
{
    prediction = 0       /*!< Prediction results */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__RESULT"></a>
 * \brief Provides methods to access result obtained with the compute() method of the neural networks prediction algorithm
 */
class Result : public daal::algorithms::Result
{
public:
    DAAL_CAST_OPERATOR(Result);

    Result() : daal::algorithms::Result(1) {};

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultId id) const
    {
        using namespace data_management;
        return services::staticPointerCast<Tensor, SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of neural networks model based prediction
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const data_management::TensorPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Registers user-allocated memory to store partial results of the neural networks model based prediction
     * \param[in] input Pointer to an object containing %input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of the neural networks prediction
     */
    template<typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input * >(input);

        services::SharedPtr<ForwardLayers> layers = in->get(model)->getLayers();
        services::SharedPtr<layers::forward::Result> lastLayerResult = layers->get(layers->size() - 1)->getLayerResult();
        services::Collection<size_t> resultDimensions = lastLayerResult->get(layers::forward::value)->getDimensions();
        resultDimensions[0] = in->get(data)->getDimensions().get(0);

        set(prediction::prediction, data_management::TensorPtr(
                new data_management::HomogenTensor<algorithmFPType>(resultDimensions, data_management::Tensor::doAllocate)));
    }

    /**
     * Checks result of the neural networks algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        data_management::TensorPtr predictionObject = get(neural_networks::prediction::prediction);
        if(predictionObject.get() == NULL) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
    }

    /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_PREDICTION_RESULT_ID; }

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
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

/** @} */
}
}
}
} // namespace daal
#endif
