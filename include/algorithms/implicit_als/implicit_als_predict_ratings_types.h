/* file: implicit_als_predict_ratings_types.h */
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
//  Implementation of the classes used in the rating prediction stage
//  of the implicit ALS algorithm
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_TYPES_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
/**
 * @defgroup implicit_als_prediction Prediction
 * \copydoc daal::algorithms::implicit_als::prediction
 * @ingroup implicit_als
 * @{
 */
/**
 * \brief Contains classes for making implicit ALS model-based prediction
 */
namespace prediction
{
/**
 * \brief Contains classes for computing ratings based on the implicit ALS model
 */
namespace ratings
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__METHOD"></a>
 * Available methods for computing the results of the implicit ALS model-based prediction
 */
enum Method
{
    defaultDense = 0,       /*!< Default: predicts ratings based on the ALS model and input data in the dense format */
    allUsersAllItems = 0    /*!< Predicts ratings for all users and items based on the ALS model and input data in the dense format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__MODELINPUTID"></a>
 * Available identifiers of input model objects for the rating prediction stage
 * of the implicit ALS algorithm
 */
enum ModelInputId
{
    model = 0           /*!< %Input model trained by the ALS algorithm */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__PARTIALMODELINPUTID"></a>
 * Available identifiers of input PartialModel objects for the rating prediction stage
 * of the implicit ALS algorithm
 */
enum PartialModelInputId
{
    usersPartialModel = 0,      /*!< %Input partial model with users factors trained by the implicit ALS algorithm
                                     in the distributed processing mode */
    itemsPartialModel = 1       /*!< %Input partial model with items factors trained by the implicit ALS algorithm
                                     in the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__PARTIALRESULTID"></a>
 * Available identifiers of input PartialModel objects for the rating prediction stage
 * of the implicit ALS algorithm
 */
enum PartialResultId
{
    finalResult = 0             /*!< Result of the implicit ALS ratings prediction algorithm */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RESULTID"></a>
 * Available identifiers of the results of the rating prediction stage of the implicit ALS algorithm
 */
enum ResultId
{
    prediction = 0        /*!< Numeric table with the predicted ratings */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__INPUTIFACE"></a>
 * \brief %Input interface for the rating prediction stage of the implicit ALS algorithm
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    virtual ~InputIface() {}

    /**
     * Returns the number of rows in the input numeric table
     * \return Number of rows in the input numeric table
     */
    virtual size_t getNumberOfUsers()    const = 0;

    /**
     * Returns the number of columns in the input numeric table
     * \return Number of columns in the input numeric table
     */
    virtual size_t getNumberOfItems() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__INPUT"></a>
 * \brief %Input objects for the rating prediction stage of the implicit ALS algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    virtual ~Input() {}

    /**
     * Returns an input Model object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<Model> get(ModelInputId id) const;

    /**
     * Sets an input Model object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const services::SharedPtr<Model> &ptr);

    /**
     * Returns the number of rows in the input numeric table
     * \return Number of rows in the input numeric table
     */
    size_t getNumberOfUsers() const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of columns in the input numeric table
     * \return Number of columns in the input numeric table
     */
    size_t getNumberOfItems() const DAAL_C11_OVERRIDE;

    /**
     * Checks the input objects and parameters of the implicit ALS algorithm in the rating prediction stage
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the rating prediction stage of the implicit ALS algorithm
 * in the distributed processing mode
 */
template<ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the first step of the rating prediction stage of the implicit ALS algorithm
 * in the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public InputIface
{
public:
    DistributedInput();

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(PartialModelInputId id) const;

    /**
     * Sets an input object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const services::SharedPtr<PartialModel> &ptr);

    /**
     * Returns the number of rows in the input numeric table
     * \return Number of rows in the input numeric table
     */
    size_t getNumberOfUsers() const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of columns in the input numeric table
     * \return Number of columns in the input numeric table
     */
    size_t getNumberOfItems() const DAAL_C11_OVERRIDE;

    /**
     * Checks the parameters of the rating prediction stage of the implicit ALS algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RESULT"></a>
 * \brief Provides methods to access the prediction results obtained with the compute() method
 *        of the implicit ALS algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();
    virtual ~Result()
    {}

    /**
     * Returns the prediction result of the implicit ALS algorithm
     * \param[in] id   Identifier of the prediction result, \ref ResultId
     * \return         Prediction result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the prediction result of the implicit ALS algorithm
     * \param[in] id    Identifier of the prediction result, \ref ResultId
     * \param[in] ptr   Pointer to the prediction result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Allocates memory to store the result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks the result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     *  Deserializes an object
     *  \param[in]  arch  Storage for a deserialized object or data structure
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS initialization algorithm in the rating prediction stage */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    PartialResult();
    /** Default destructor */
    virtual ~PartialResult() {}

    /**
     * Allocates memory to store partial results of the rating prediction stage of the implicit ALS algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns a partial result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<Result> get(PartialResultId id) const;

    /**
     * Sets a partial result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(PartialResultId id, const services::SharedPtr<Result> &ptr);

    /**
     * Checks a partial result of the implicit ALS algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
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
};

} // interface1
using interface1::InputIface;
using interface1::Input;
using interface1::DistributedInput;
using interface1::PartialResult;
using interface1::Result;

}
}
/** @} */
}
}
}

#endif
