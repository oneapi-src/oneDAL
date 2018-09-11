/* file: logistic_regression_predict_types.h */
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

#ifndef __LOGISTIC_REGRESSION_PREDICT_TYPES_H__
#define __LOGISTIC_REGRESSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
/**
 * @defgroup logistic_regression_prediction Prediction
 * \copydoc daal::algorithms::logistic_regression::prediction
 * @ingroup logistic regression
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model */
namespace prediction
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__METHOD"></a>
 * Available methods for predictions based on the logistic regression model
 */
enum Method
{
    defaultDense = 0        /*!< Default method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__RESULTTOCOMPUTEID"></a>
* Available identifiers to specify the result to compute
*/
enum ResultToComputeId
{
    computeClassesLabels = 0x00000001ULL,
    computeClassesProbabilities = 0x00000002ULL,
    computeClassesLogProbabilities = 0x00000004ULL
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__RESULT_NUMERIC_TABLE_ID"></a>
* Available identifiers of results obtained in the prediction stage of the classification algorithm
*/
enum ResultNumericTableId
{
    probabilities = classifier::prediction::lastResultId + 1, /*!< Numeric table of size: n x 1, if nClasses = 2, n x nClasses, if nClasses > 2
                                                                   containing probabilities of classes computed when
                                                                   computeClassesProbabilities option is enabled.
                                                                   In case  nClasses = 2 the table contains probabilities of class _1. */
    logProbabilities,                                         /*!< Numeric table of size: n x 1, if nClasses = 2, n x nClasses, if nClasses > 2
                                                                   containing logarithms of classes_ probabilities computed when
                                                                   computeClassesLogProbabilities option is enabled.
                                                                   In case nClasses = 2 the table contains logarithms of class _1_ probabilities. */
    lastResultNumericTableId = logProbabilities
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__PARAMETER"></a>
 * \brief Parameters of the prediction algorithm
 *
 * \snippet logistic_regression/logistic_regression_predict_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    Parameter(size_t nClasses = 2) : daal::algorithms::classifier::Parameter(nClasses), resultsToCompute(computeClassesLabels) {}
    Parameter(const Parameter& o) : daal::algorithms::classifier::Parameter(o), resultsToCompute(o.resultsToCompute){}
    DAAL_UINT64 resultsToCompute;           /*!< 64 bit integer flag that indicates the results to compute */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the LOGISTIC_REGRESSION algorithm
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;
public:
    Input() : super(){}
    Input(const Input& other) : super(other){}
    virtual ~Input() {}

    using super::get;
    using super::set;

    /**
     * Returns the input Numeric Table object in the prediction stage of the algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(classifier::prediction::NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the LOGISTIC_REGRESSION algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    logistic_regression::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets the input Model object in the prediction stage of the LOGISTIC_REGRESSION algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const logistic_regression::ModelPtr &ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__RESULT"></a>
* \brief Provides interface for the result of model-based prediction
*/
class DAAL_EXPORT Result : public algorithms::classifier::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
    * Returns the result of model-based prediction
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(classifier::prediction::ResultId id) const;

    /**
    * Sets the result of model-based prediction
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(classifier::prediction::ResultId id, const data_management::NumericTablePtr &value);

    /**
    * Returns the result of model-based prediction
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
    * Sets the result of model-based prediction
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr &value);

    /**
    * Allocates memory to store a partial result of model-based prediction
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Algorithm method
    * \return Status of allocation
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
    * Checks the result of model-based prediction
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::classifier::prediction::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;

} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;
}
/** @} */
}
}
}
#endif // __LOGISTIC_REGRESSION_PREDICT_TYPES_H__
