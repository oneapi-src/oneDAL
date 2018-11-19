/* file: linear_regression_predict_types.h */
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
//  Implementation of the linear regression algorithm interface
//--
*/

#ifndef __LINEAR_REGRESSION_PREDICT_TYPES_H__
#define __LINEAR_REGRESSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/linear_regression/linear_regression_model.h"
#include "algorithms/linear_model/linear_model_predict_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the linear regression algorithm
 */
namespace linear_regression
{
/**
 * @defgroup linear_regression_prediction Prediction
 * \copydoc daal::algorithms::linear_regression::prediction
 * @ingroup linear_regression
 * @{
 */
/**
 * \brief Contains a class for making linear regression model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__METHOD"></a>
 * \brief Available methods for making linear regression model-based prediction
 */
enum Method
{
    defaultDense = 0
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making linear regression model-based prediction
 */
enum NumericTableInputId
{
    data = linear_model::prediction::data, /*!< Input data table */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making linear regression model-based prediction
 */
enum ModelInputId
{
    model = linear_model::prediction::model, /*!< Trained linear regression model */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making linear regression model-based prediction
 */
enum ResultId
{
    prediction = linear_model::prediction::prediction, /*!< Result of linear regression model-based prediction */
    lastResultId = prediction
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making linear regression model-based prediction
 */
class DAAL_EXPORT Input : public linear_model::prediction::Input
{
public:
    /** Default constructor */
    Input();

    /**
     * Returns an input object for making linear regression model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns an input object for making linear regression model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    linear_regression::ModelPtr get(ModelInputId id) const;

    /**
     * Sets an input object for making linear regression model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr &value);

    /**
     * Sets an input object for making linear regression model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ModelInputId id, const linear_regression::ModelPtr &value);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of linear regression model-based prediction
 */
class DAAL_EXPORT Result : public linear_model::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    /**
     * Returns the result of linear regression model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of linear regression model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;
}
/** @} */
}
}
} // namespace daal
#endif
