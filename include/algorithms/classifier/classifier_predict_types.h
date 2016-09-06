/* file: classifier_predict_types.h */
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
//  Implementation of the base classes used in the prediction stage
//  of the classifier algorithm
//--
*/

#ifndef __CLASSIFIER_PREDICT_TYPES_H__
#define __CLASSIFIER_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_model.h"

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
/**
 * @defgroup prediction Prediction
 * \copydoc daal::algorithms::classifier::prediction
 * @ingroup classifier
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__PREDICTION__NUMERICTABLEINPUTID"></a>
 * Available identifiers of input NumericTable objects in the prediction stage
 * of the classification algorithm
 */
enum NumericTableInputId
{
    data = 0        /*!< Input data set */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__PREDICTION__MODELINPUTID"></a>
 * Available identifiers of input Model objects in the prediction stage
 * of the classification algorithm
 */
enum ModelInputId
{
    model = 1       /*!< Input model trained by the classification algorithm */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__PREDICTION__RESULTID"></a>
 * Available identifiers of results obtained in the prediction stage of the classification algorithm
 */
enum ResultId
{
    prediction = 0       /*!< Prediction results */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__INPUTIFACE"></a>
 * \brief Base class for working with input objects in the prediction stage of the classification algorithm
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    virtual ~InputIface() {}
    /**
     * Returns the number of rows in the input data set
     * \return Number of rows in the input data set
     */
    virtual size_t getNumberOfRows() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the classification algorithm
 */
class Input : public InputIface
{
public:
    Input() : InputIface(2) {}
    virtual ~Input() {}

    /**
     * Returns the number of rows in the input data set
     * \return Number of rows in the input data set
     */
    size_t getNumberOfRows() const DAAL_C11_OVERRIDE
    {
        size_t nRows = 0;
        data_management::NumericTablePtr dataTable = get(data);
        if(dataTable)
        {
            nRows = dataTable->getNumberOfRows();
        }
        else
        {
            /* ERROR */;
        }
        return nRows;
    }

    /**
     * Returns the input Numeric Table object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the input Model object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<classifier::Model> get(ModelInputId id) const
    {
        return services::staticPointerCast<classifier::Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets the input Model object in the prediction stage of the classifier algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const services::SharedPtr<Model> &ptr)
    {
        Argument::set(id, ptr);
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
        if(parameter != NULL)
        {
            const Parameter *algParameter = static_cast<const Parameter *>(parameter);
            if (algParameter->nClasses < 2) { this->_errors->add(services::ErrorIncorrectNumberOfClasses); return; }
        }

        data_management::NumericTablePtr dataTable = get(data);
        if(!data_management::checkNumericTable(dataTable.get(), this->_errors.get(), dataStr())) { return; }

        services::SharedPtr<classifier::Model> m = get(model);
        if(!m) { this->_errors->add(services::ErrorNullModel); return; }

        size_t trainingDataFeatures = m->getNFeatures();
        size_t predictionDataFeatures = dataTable->getNumberOfColumns();
        if(trainingDataFeatures != predictionDataFeatures)
        {
            services::ErrorPtr error(new services::Error(services::ErrorIncorrectNumberOfColumns));
            error->addStringDetail(services::ArgumentName, dataStr());
            this->_errors->add(error);
            return;
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__RESULT"></a>
 * \brief Provides methods to access prediction results obtained with the compute() method
 *        of the classifier prediction algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    /**
     * Returns the prediction result of the classification algorithm
     * \param[in] id   Identifier of the prediction result, \ref ResultId
     * \return         Prediction result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the prediction result of the classification algorithm
     * \param[in] id    Identifier of the prediction result, \ref ResultId
     * \param[in] value Pointer to the prediction result
     */
    void set(ResultId id, const data_management::NumericTablePtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Allocates memory for storing prediction results of the classification algorithm
     * \tparam  algorithmFPType     Data type for storing prediction results
     * \param[in] input     Pointer to the input objects of the classification algorithm
     * \param[in] parameter Pointer to the parameters of the classification algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        set(prediction, data_management::NumericTablePtr(
                new data_management::HomogenNumericTable<algorithmFPType>(
                    1, (static_cast<const InputIface *>(input))->getNumberOfRows(),
                    data_management::NumericTableIface::doAllocate)));
    }

    /**
     * Checks the correctness of the Result object
     * \param[in] input     Pointer to the the input object
     * \param[in] parameter Pointer to the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        checkImpl(input, parameter);
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_CLASSIFIER_PREDICTION_RESULT_ID; }

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
        size_t nRows = (static_cast<const InputIface *>(input))->getNumberOfRows();
        data_management::NumericTablePtr resTable = get(prediction);

        int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray;
        if(!data_management::checkNumericTable(resTable.get(), this->_errors.get(), predictionStr(), unexpectedLayouts, 0, 1, nRows)) { return; }
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::Result;
}
/** @} */
}
}
}
#endif
