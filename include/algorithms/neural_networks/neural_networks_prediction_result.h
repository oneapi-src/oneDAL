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
#include "services/collection.h"
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
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__RESULTID"></a>
 * Available identifiers of results obtained in the prediction stage of the neural network algorithm
 */
enum ResultCollectionId
{
    predictionCollection = 1       /*!< Prediction results */
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

    Result() : daal::algorithms::Result(2)
    {
        set(predictionCollection, data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
    }

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultId id) const
    {
        using namespace data_management;
        KeyValueDataCollectionPtr collection = get(predictionCollection);
        if (!collection) { return TensorPtr(); }
        if (collection->size() == 0) { return TensorPtr(); }
        return Tensor::cast(collection->getValueByIndex(0));
    }

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(ResultCollectionId id) const
    {
        using namespace data_management;
        return services::staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the result of the neural networks model based prediction
     * \param[in] id    Identifier of the result
     * \param[in] key   Index of the tensor with partial results in the key-value data collection
     * \return          Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(ResultCollectionId id, size_t key) const
    {
        using namespace data_management;
        KeyValueDataCollectionPtr collection = get(id);
        if (!collection) { return TensorPtr(); }
        return services::staticPointerCast<Tensor, SerializationIface>((*collection)[key]);
    }

    /**
     * Sets the result of neural networks model based prediction
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const data_management::TensorPtr &value)
    {
        add(predictionCollection, 0, value);
    }

    /**
     * Sets the result of neural networks model based prediction
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultCollectionId id, const data_management::KeyValueDataCollectionPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Add the value to the key-value data collection of partial results
     * \param[in] id    Identifier of the result
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Result
     */
    void add(ResultCollectionId id, size_t key, const data_management::TensorPtr &value)
    {
        data_management::KeyValueDataCollectionPtr collection = get(id);
        if (!collection) { return; }
        (*collection)[key] = value;
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
        using namespace data_management;
        const Input *in = static_cast<const Input * >(input);

        services::SharedPtr<Model> predictionModel = in->get(model);
        const services::Collection<size_t> &dataSize = in->get(data)->getDimensions();

        predictionModel->allocate<algorithmFPType>(dataSize, parameter);

        services::SharedPtr<ForwardLayers> layers = predictionModel->getLayers();
        services::SharedPtr<services::Collection<layers::NextLayers> > nextLayers = predictionModel->getNextLayers();
        size_t nLayers = layers->size();
        services::Collection<size_t> lastLayerIds;
        for (size_t layerId = 0; layerId < nLayers; layerId++)
        {
            if (nextLayers->get(layerId).size() == 0)
            {
                lastLayerIds.push_back(layerId);
            }
        }

        size_t nLastLayers = lastLayerIds.size();
        size_t nResults = in->get(data)->getDimensionSize(0);

        for (size_t i = 0; i < nLastLayers; i++)
        {
            size_t layerId = lastLayerIds[i];
            services::SharedPtr<layers::forward::Result> lastLayerResult = layers->get(layerId)->getLayerResult();
            services::Collection<size_t> resultDimensions = lastLayerResult->get(layers::forward::value)->getDimensions();
            resultDimensions[0] = in->get(data)->getDimensions().get(0);

            add(prediction::predictionCollection, layerId, TensorPtr(
                    new HomogenTensor<algorithmFPType>(resultDimensions, Tensor::doAllocate)));
        }
    }

    /**
     * Checks result of the neural networks algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        using namespace daal::data_management;
        using namespace daal::services;
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        KeyValueDataCollectionPtr predictionTensorCollection = get(predictionCollection);
        if (!predictionTensorCollection)
        { this->_errors->add(ErrorNullOutputDataCollection); return; }

        if (predictionTensorCollection->size() == 0)
        { this->_errors->add(ErrorIncorrectNumberOfElementsInResultCollection); return; }

        const prediction::Input *algInput = static_cast<const prediction::Input *>(input);
        TensorPtr dataTensor = algInput->get(prediction::data);

        size_t nSamples = dataTensor->getDimensionSize(0);

        size_t nLastLayers = predictionTensorCollection->size();
        for (size_t i = 0; i < nLastLayers; i++)
        {
            size_t layerId = predictionTensorCollection->getKeyByIndex((int)i);
            TensorPtr predictionTensor = get(predictionCollection, layerId);
            if (!predictionTensor)
            {
                SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorNullTensor));
                error->addStringDetail(ArgumentName, predictionStr());
                error->addIntDetail(ElementInCollection, (int)layerId);
                this->_errors->add(error);
                return;
            }
            Collection<size_t> expectedDims = predictionTensor->getDimensions();
            expectedDims[0] = nSamples;
            if (!checkTensor(predictionTensor.get(), this->_errors.get(), predictionStr(), &expectedDims)) { return; }
        }
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
