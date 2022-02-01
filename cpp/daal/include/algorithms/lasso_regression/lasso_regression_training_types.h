/* file: lasso_regression_training_types.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the lasso regression algorithm interface
//--
*/

#ifndef __LASSO_REGRESSION_TRAINING_TYPES_H__
#define __LASSO_REGRESSION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/lasso_regression/lasso_regression_model.h"
#include "algorithms/linear_model/linear_model_training_types.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the lasso regression algorithm
 */
namespace lasso_regression
{
/**
 * @defgroup lasso_regression_training Training
 * \copydoc daal::algorithms::lasso_regression::training
 * @ingroup lasso_regression
 * @{
 */
/**
 * \brief Contains a class for lasso regression model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LASSO_REGRESSION__TRAINING__METHOD"></a>
 * \brief Computation methods for lasso regression model-based training
 */
enum Method
{
    defaultDense = 0 /*!< Normal equations method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LASSO_REGRESSION__TRAINING__INPUTID"></a>
 * \brief Available identifiers of input objects for lasso regression model-based training
 */
enum InputId
{
    data               = linear_model::training::data,               /*!< %Input data table */
    dependentVariables = linear_model::training::dependentVariables, /*!< Values of the dependent variable for the input data */
    lastInputId        = dependentVariables
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALINPUTID"></a>
* Available identifiers of optional input for the iterative solver
*/
enum OptionalInputId
{
    optionalArgument    = lastInputId + 1, /*!< Algorithm-specific input data, can be generated by previous runs of the algorithm */
    lastOptionalInputId = optionalArgument
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALDATAID"></a>
* Available identifiers of optional input for the iterative solver
*/
enum OptionalDataId
{
    weights,    /*!< NumericTable of size 1 x n with weights of samples. Applied for all method */
    gramMatrix, /*!< NumericTable of size p x p with last iteration number. Applied for all method */
    lastOptionalData = gramMatrix
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LASSO_REGRESSION__TRAINING__OPTIONALRESULTTOCOMPUTEID"></a>
* Available identifiers to specify the result to compute
*/
enum ResultToComputeId
{
    computeGramMatrix = 0x00000001ULL /*!< The flag to compute Gram Matrix */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LASSO_REGRESSION__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the result of lasso regression model-based training
 */
enum ResultId
{
    model        = linear_model::training::model, /*!< Lasso regression model */
    lastResultId = model
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LASSO_REGRESSION__TRAINING__RESULT_NUMERIC_TABLE_ID"></a>
* Available identifiers of results obtained in the training stage of the regression algorithm
*/
enum OptionalResultNumericTableId
{
    gramMatrixId             = lastResultId + 1, /*!< Numeric table of size: p x p, containing computed Gram matrix */
    lastResultNumericTableId = gramMatrixId
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LASSO_REGRESSION__TRAINING__RESULTID"></a>
 * \brief Available identifiers for input data corruption
 */
enum DataUseInComputation
{
    doNotUse = 0, /*!< The input data and labels can`t be corrupted */
    doUse    = 1  /*!< The input data and labels can be corrupted */
};
/**
 * \brief Contains version 1.0 of the Intel(R) oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LASSO_REGRESSION__TRAINPARAMETER"></a>
 * \brief Parameters for the lasso regression algorithm
 *
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public linear_model::Parameter
{
    typedef optimization_solver::iterative_solver::BatchPtr SolverPtr;

    Parameter(const SolverPtr & solver = SolverPtr());
    Parameter(const Parameter & o)
        : linear_model::Parameter(o),
          lassoParameters(o.lassoParameters),
          optimizationSolver(o.optimizationSolver),
          dataUseInComputation(o.dataUseInComputation),
          optResultToCompute(o.optResultToCompute)
    {}

    services::Status check() const DAAL_C11_OVERRIDE;

    data_management::NumericTablePtr lassoParameters; /*!< Numeric table that contains values of lasso parameters */

    SolverPtr optimizationSolver; /*!< Default is coordinate descent solver */

    DataUseInComputation dataUseInComputation; /*!< The flag allows to corrupt input data */
    DAAL_UINT64 optResultToCompute;            /*!< 64 bit integer flag that indicates the optional results to compute */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__TRAINING__INPUTIFACE"></a>
 * \brief Abstract class that specifies the interface of input objects for lasso regression model-based training
 */
class InputIface
{
public:
    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    virtual size_t getNumberOfFeatures() const = 0;

    /**
     * Returns the number of dependent variables
     * \return Number of dependent variables
     */
    virtual size_t getNumberOfDependentVariables() const = 0;

    virtual ~InputIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__TRAINING__INPUT"></a>
 * \brief %Input objects for lasso regression model-based training
 */
class DAAL_EXPORT Input : public linear_model::training::Input, public InputIface
{
public:
    /** Default constructor */
    Input();
    Input(const Input & other);

    virtual ~Input() {}

    /**
     * Returns an input object for lasso regression model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for lasso regression model-based training
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
    * Returns optional input of the iterative solver algorithm
    * \param[in] id    Identifier of the optional input data
    * \return          %Input data that corresponds to the given identifier
    */
    algorithms::OptionalArgumentPtr get(OptionalInputId id) const;

    /**
    * Sets optional input for the iterative solver algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalInputId id, const algorithms::OptionalArgumentPtr & ptr);

    /**
    * Returns input NumericTable containing optional data
    * \param[in] id    Identifier of the input numeric table
    * \return          %Input numeric table that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
    * Sets optional input for the algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Returns the number of dependent variables
    * \return Number of dependent variables
    */
    size_t getNumberOfDependentVariables() const DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for the lasso regression algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of lasso regression model-based training
 */
class DAAL_EXPORT Result : public linear_model::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of lasso regression model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    lasso_regression::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of lasso regression model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const lasso_regression::ModelPtr & value);

    /**
    * Returns the result of lasso regression model-based training
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalResultNumericTableId id) const;

    /**
    * Sets the result of lasso regression model-based training
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(OptionalResultNumericTableId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store the result of lasso regression model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of lasso regression model-based training
     * \param[in] method Computation method for the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method);

    /**
     * Checks the result of lasso regression model-based training
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;
} // namespace interface1

using interface1::InputIface;
using interface1::Input;
using interface1::Parameter;

using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;

} // namespace training
} // namespace lasso_regression
/** @} */
} // namespace algorithms
} // namespace daal

#endif
