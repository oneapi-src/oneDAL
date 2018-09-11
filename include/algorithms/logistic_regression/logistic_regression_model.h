/* file: logistic_regression_model.h */
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
//  Implementation of logistic regression model class
//--
*/

#ifndef __LOGISTIC_REGRESIION_MODEL_H__
#define __LOGISTIC_REGRESIION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup logistic_regression Logistic regression
 * \copydoc daal::algorithms::logistic_regression
 */
namespace logistic_regression
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup logistic_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__MODEL"></a>
 * \brief %Model of the classifier trained by the logistic_regression::training::Batch algorithm.
 *
 * \par References
 *      - \ref logistic_regression::training::interface1::Batch "training::Batch" class
 *      - \ref logistic_regression::prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    /**
    * Returns the number of regression coefficients
    * \return Number of regression coefficients
    */
    virtual size_t getNumberOfBetas() const = 0;

    /**
    * Returns true if the regression model contains the intercept term, and false otherwise
    * \return True if the regression model contains the intercept term, and false otherwise
    */
    virtual bool getInterceptFlag() const = 0;

    /**
    * Returns the numeric table that contains regression coefficients
    * \return Table that contains regression coefficients
    */
    virtual data_management::NumericTablePtr getBeta() = 0;

    /**
    * Returns the numeric table that contains regression coefficients
    * \return Table that contains regression coefficients
    */
    virtual const data_management::NumericTablePtr getBeta() const = 0;

protected:
    Model() : classifier::Model()
    {}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif
