/* file: logistic_regression_training_types.h */
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
//  Implementation of logistic regression training algorithm interface.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAINING_TYPES_H__
#define __LOGISTIC_REGRESSION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
/**
 * @defgroup logistic_regression_training Training
 * \copydoc daal::algorithms::logistic_regression::training
 * @ingroup logistic regression
 * @{
 */
/**
 * \brief Contains classes for logistic regression models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__METHOD"></a>
 * \brief Computation methods for logistic regression model-based training
 */
enum Method
{
    defaultDense = 0  /*!< Default training method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__PARAMETER"></a>
 * \brief logistic regression algorithm parameters
 *
 * \snippet logistic_regression/logistic_regression_training_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    typedef optimization_solver::iterative_solver::BatchPtr SolverPtr;

    /** Default constructor */
    Parameter(size_t nClasses, const SolverPtr& solver = SolverPtr());
    Parameter(const Parameter& o) : classifier::Parameter(o), interceptFlag(o.interceptFlag),
        penaltyL1(o.penaltyL1), penaltyL2(o.penaltyL2), optimizationSolver(o.optimizationSolver){}
    services::Status check() const DAAL_C11_OVERRIDE;

    bool interceptFlag;  /*!< Whether the intercept needs to be computed */
    float penaltyL1;     /*!< L1 regularization coefficient. Default is 0 (not applied) */
    float penaltyL2;     /*!< L2 regularization coefficient. Default is 0 (not applied) */
    SolverPtr optimizationSolver; /*!< Default is sgd momentum solver */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of model-based training
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the logistic regression algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the logistic regression algorithm
     */
    ModelPtr get(classifier::training::ResultId id) const;

    /**
    * Sets the result of model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(classifier::training::ResultId id, const ModelPtr &value);

    /**
     * Allocates memory to store final results of the logistic regression training algorithm
     * \param[in] input         %Input of the logistic regression training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        logistic regression computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**s
    * Checks the result of model-based training
    * \param[in] input   %Input object for the algorithm
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
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Parameter;
using interface1::Result;
using interface1::ResultPtr;

} // namespace daal::algorithms::logistic_regression::training
/** @} */
}
}
} // namespace daal
#endif // __LOGISTIC_REGRESSION_TRAINING_TYPES_H__
