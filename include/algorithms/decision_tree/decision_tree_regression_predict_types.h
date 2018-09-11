/* file: decision_tree_regression_predict_types.h */
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
//  Implementation of the Decision tree regression algorithm interface
//--
*/

#ifndef __DECISION_TREE_REGRESSION_PREDICT_TYPES_H__
#define __DECISION_TREE_REGRESSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/decision_tree/decision_tree_regression_model.h"
#include "algorithms/regression/regression_predict_types.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains classes of the Decision tree algorithm
 */
namespace decision_tree
{

/**
 * \brief Contains classes of the Decision tree regression algorithm
 */
namespace regression
{

/**
 * @defgroup decision_tree_regression_prediction Prediction
 * \copydoc daal::algorithms::decision_tree::regression::prediction
 * @ingroup decision_tree_regression
 * @{
 */
/**
 * \brief Contains a class for making Decision tree model-based prediction
 */
namespace prediction
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__METHOD"></a>
 * \brief Available methods for making Decision tree model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making decision tree model-based prediction
 */
enum NumericTableInputId
{
    data = algorithms::regression::prediction::data, /*!< Input data table */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PREDICTION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making decision tree model-based prediction
 */
enum ModelInputId
{
    model = algorithms::regression::prediction::model, /*!< Trained decision tree model */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making decision tree model-based prediction
 */
enum ResultId
{
    prediction = algorithms::regression::prediction::prediction, /*!< Result of decision tree model-based prediction */
    lastResultId = prediction
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making Decision tree model-based prediction
 */
class DAAL_EXPORT Input : public algorithms::regression::prediction::Input
{
public:
    Input();
    Input(const Input &other);

    /**
     * Returns an input object for making Decision tree model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the Decision tree algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the regression algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the Decision tree algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input Model object
     */
    void set(ModelInputId id, const ModelPtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of decision tree model-based prediction
 */
class DAAL_EXPORT Result : public algorithms::regression::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    /**
     * Returns the result of decision tree model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of decision tree model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory to store a partial result of decision tree model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Checks the result of decision tree model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

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

} // namespace prediction
/** @} */
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
