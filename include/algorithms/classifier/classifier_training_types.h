/* file: classifier_training_types.h */
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
//  Implementation of the base classes used in the training stage
//  of the classification algorithms
//--
*/

#ifndef __CLASSIFIER_TRAINING_TYPES_H__
#define __CLASSIFIER_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup classifier Base Classifier
 * \copydoc daal::algorithms::classifier
 * @ingroup classification
 */
/**
 * \brief Contains classes for working with classifiers
 */
namespace classifier
{
/**
 * @defgroup training Training
 * \copydoc daal::algorithms::classifier::training
 * @ingroup classifier
 * @{
 */
/**
 * \brief Contains classes for training the model of the classification algorithms
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__INPUTID"></a>
 * Available identifiers of the results in the training stage of the classification algorithms
 */
enum InputId
{
    data     = 0,        /*!< Training data set */
    labels   = 1,        /*!< Labels of the training data set */
    weights  = 2         /*!< Optional. Weights of the observations in the training data set */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__PARTIALRESULTID"></a>
 * Available identifiers of partial results
 */
enum PartialResultId
{
    partialModel = 0        /*!< Trained partial model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__RESULTID"></a>
 * Available identifiers of the results in the training stage of the classification algorithms
 */
enum ResultId
{
    model = 0           /*!< Resulting model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__STEP2MASTERINPUTID"></a>
 * Available identifiers of the step 2 input
 */
enum Step2MasterInputId
{
    partialModels = 0        /*!< Collection of partial models trained on local nodes */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__INPUTIFACE"></a>
 * \brief Abstract class that specifies the interface of the classes of the classification algorithm input objects
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    virtual ~InputIface() {}
    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__INPUT"></a>
 * \brief Base class for the input objects in the training stage of the classification algorithms
 */
class Input : public InputIface
{
public:
    Input() : InputIface(3) {}

    virtual ~Input() {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        return get(classifier::training::data)->getNumberOfColumns();
    }

    /**
     * Returns the input object in the training stage of the classification algorithm
     * \param[in] id   Identifier of the input object, \ref InputId
     * \return         Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input object in the training stage of the classification algorithm
     * \param[in] id    Identifier of the input object, \ref InputId
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        checkImpl(parameter);
    }

protected:

    void checkImpl(const daal::algorithms::Parameter *parameter) const
    {
        if (parameter != NULL)
        {
            const Parameter *algParameter = static_cast<const Parameter *>(parameter);
            if (algParameter->nClasses < 2)
            {
                this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
                return;
            }
        }

        data_management::NumericTablePtr dataTable = get(data);
        if(!data_management::checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

        size_t nRows = dataTable->getNumberOfRows();

        data_management::NumericTablePtr labelsTable = get(labels);
        if(!data_management::checkNumericTable(labelsTable.get(), this->_errors.get(), labelsStr(), 0, 0, 1, nRows)) { return; }

        data_management::NumericTablePtr weightsTable = get(weights);
        if(weightsTable)
        {
            if(!data_management::checkNumericTable(weightsTable.get(), this->_errors.get(), weightsStr(), 0, 0, 1, nRows)) { return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the classifier training algorithm
 *        in the online or distributed processing mode
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult() : daal::algorithms::PartialResult(1) {};
    virtual ~PartialResult() {}

    /**
     * Returns the partial result in the training stage of the classification algorithm
     * \param[in] id   Identifier of the partial result, \ref PartialResultId
     * \return         Partial result that corresponds to the given identifier
     */
    services::SharedPtr<classifier::Model> get(PartialResultId id) const
    {
        return services::staticPointerCast<classifier::Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the partial result in the training stage of the classification algorithm
     * \param[in] id    Identifier of the partial result, \ref PartialResultId
     * \param[in] value Pointer to the partial result
     */
    void set(PartialResultId id, const services::SharedPtr<daal::algorithms::classifier::Model> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks the correctness of the PartialResult object
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        checkImpl(input, parameter);
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_CLASSIFIER_TRAINING_PARTIAL_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }

    void checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
    {
        services::SharedPtr<daal::algorithms::classifier::Model> m = get(partialModel);
        if(!m) { this->_errors->add(services::ErrorNullModel); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method in the
 *        batch processing mode or finalizeCompute() method
 *        in the online or distributed processing mode of the classification algorithm
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {}
    virtual ~Result() {}

    /**
     * Returns the model trained with the classification algorithm
     * \param[in] id    Identifier of the result, \ref ResultId
     * \return          Model trained with the classification algorithm
     */
    services::SharedPtr<daal::algorithms::classifier::Model> get(ResultId id) const
    {
        return services::staticPointerCast<daal::algorithms::classifier::Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the training stage of the classification algorithm
     * \param[in] id    Identifier of the result, \ref ResultId
     * \param[in] value Pointer to the training result
     */
    void set(ResultId id, const services::SharedPtr<daal::algorithms::classifier::Model> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks the correctness of the Result object
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        checkImpl(input, parameter);
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_CLASSIFIER_TRAINING_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    void checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
    {
        services::SharedPtr<daal::algorithms::classifier::Model> m = get(model);
        if(!m) { this->_errors->add(services::ErrorNullModel); return; }
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief Input objects of the classification training algorithm in the distributed processing mode
 */
class DistributedInput : public InputIface
{
public:
    DistributedInput() : InputIface(1)
    {
        Argument::set(partialModels, data_management::DataCollectionPtr(new data_management::DataCollection()));
    }

    virtual ~DistributedInput() {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        data_management::DataCollectionPtr models = get(classifier::training::partialModels);
        classifier::Model *firstModel =
            static_cast<classifier::Model *>((*models)[0].get());
        return firstModel->getNFeatures();
    }

    /**
     * Returns input objects of the classification algorithm in the distributed processing mode
     * \param[in] id    Identifier of the input objects
     * \return          Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Adds input object on the master node in the training stage of the classification algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] partialResult Pointer to the object
     */
    void add(const Step2MasterInputId &id, const services::SharedPtr<PartialResult> &partialResult)
    {
        data_management::DataCollectionPtr collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
        collection->push_back(services::staticPointerCast<data_management::SerializationIface, classifier::Model>(
                                  partialResult->get(partialModel)));
    }

    /**
     * Sets input object in the training stage of the classification algorithm
     * \param[in] id   Identifier of the object
     * \param[in] value Pointer to the object
     */
    void set(Step2MasterInputId id, const data_management::DataCollectionPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks input parameters in the training stage of the classification algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Algorithm method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        data_management::DataCollectionPtr spModels = get(partialModels);
        data_management::DataCollection *models = spModels.get();
        if (models == 0) { this->_errors->add(services::ErrorNullModel); return; }

        size_t size = models->size();
        if (size == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        for (size_t i = 0; i < size; i++)
        {
            classifier::Model *model = (classifier::Model *)((*models)[i].get());
            if (model == 0) { this->_errors->add(services::ErrorNullModel); return; }
        }
    }
};
} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::Result;

} // namespace daal::algorithms::classifier::training
/** @} */
} // namespace daal::algorithms::classifier
} // namespace daal::algorithms
} // namespace daal
#endif
