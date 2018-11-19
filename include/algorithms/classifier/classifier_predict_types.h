/* file: classifier_predict_types.h */
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
    data,                      /*!< Input data set */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__PREDICTION__MODELINPUTID"></a>
 * Available identifiers of input Model objects in the prediction stage
 * of the classification algorithm
 */
enum ModelInputId
{
    model = lastNumericTableInputId + 1,               /*!< Input model trained by the classification algorithm */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__PREDICTION__RESULTID"></a>
 * Available identifiers of results obtained in the prediction stage of the classification algorithm
 */
enum ResultId
{
    prediction,           /*!< Prediction results */
    lastResultId = prediction
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
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements);
    InputIface(const InputIface& other) : daal::algorithms::Input(other){}

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
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    Input(const Input& other) : InputIface(other){}
    virtual ~Input() {}

    /**
     * Returns the number of rows in the input data set
     * \return Number of rows in the input data set
     */
    size_t getNumberOfRows() const DAAL_C11_OVERRIDE;

    /**
     * Returns the input Numeric Table object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    classifier::ModelPtr get(ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets the input Model object in the prediction stage of the classifier algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const ModelPtr &ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter *parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__RESULT"></a>
 * \brief Provides methods to access prediction results obtained with the compute() method
 *        of the classifier prediction algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    /**
     * Returns the prediction result of the classification algorithm
     * \param[in] id   Identifier of the prediction result, \ref ResultId
     * \return         Prediction result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the prediction result of the classification algorithm
     * \param[in] id    Identifier of the prediction result, \ref ResultId
     * \param[in] value Pointer to the prediction result
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory for storing prediction results of the classification algorithm
     * \tparam  algorithmFPType     Data type for storing prediction results
     * \param[in] input     Pointer to the input objects of the classification algorithm
     * \param[in] parameter Pointer to the parameters of the classification algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks the correctness of the Result object
     * \param[in] input     Pointer to the the input object
     * \param[in] parameter Pointer to the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                           int method) const DAAL_C11_OVERRIDE;

protected:
    Result(size_t n);
    services::Status checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const;

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
}
/** @} */
}
}
}
#endif
