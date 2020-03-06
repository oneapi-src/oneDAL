/* file: iterative_solver_types.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of the iterative solver algorithm types.
//--
*/

#ifndef __ITERATIVE_SOLVER_TYPES_H__
#define __ITERATIVE_SOLVER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * @defgroup iterative_solver Iterative Solver
 * \copydoc daal::algorithms::optimization_solver::iterative_solver
 * @ingroup optimization_solver
 * @{
 */
/**
 * \brief Contains classes for computing the iterative solver
 */
namespace iterative_solver
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__INPUTID"></a>
 * Available identifiers of input for the iterative solver
 */
enum InputId
{
    inputArgument, /*!< Initial value to start optimization */
    lastInputId = inputArgument
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
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULTID"></a>
 * Available identifiers of results for the iterative solver algorithm
 */
enum ResultId
{
    minimum,     /*!< Numeric table of size p x 1 with the argument */
    nIterations, /*!< Table containing the number of executed iterations */
    lastResultId = nIterations
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALRESULTID"></a>
* Available identifiers of optional results for the iterative solver algorithm
*/
enum OptionalResultId
{
    optionalResult       = lastResultId + 1, /*!< Algorithm-specific result data generated if required by parameter flag optionalResultReq */
    lastOptionalResultId = optionalResult
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALDATAID"></a>
* Available identifiers of optional input for the iterative solver
*/
enum OptionalDataId
{
    lastIteration, /*!< NumericTable of size 1 x 1 with last iteration number. Applied for all method */
    lastOptionalData = lastIteration
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__PARAMETER"></a>
 * \brief %Parameter base class for the iterative solver algorithm    \DAAL_DEPRECATED
 *
 * \snippet optimization_solver/iterative_solver/iterative_solver_types.h interface1::Parameter source code
 */
/* [interface1::Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter base class of the iterative solver algorithm
     * \param[in] function_             Objective function represented as sum of functions
     * \param[in] nIterations_          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold_    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] optionalResultReq_    Flag indicating if algorithm-specific result data generation is needed
     * \param[in] batchSize_            Batch size
     */
    Parameter(const sum_of_functions::interface1::BatchPtr & function_, size_t nIterations_ = 100, double accuracyThreshold_ = 1.0e-05,
              bool optionalResultReq_ = false, size_t batchSize_ = 1);

    /**
    * Constructs an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter(const Parameter & other);

    /**
    * Copy an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter & operator=(const Parameter & other);

    ~Parameter() DAAL_C11_OVERRIDE {}

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    sum_of_functions::interface1::BatchPtr function; /*!< Objective function represented as sum of functions */
    size_t nIterations;                              /*!< Maximal number of iterations of the algorithm */
    double accuracyThreshold;                        /*!< Accuracy of the algorithm. The algorithm terminates when
                                                                this accuracy is achieved */
    bool optionalResultRequired;                     /*!< Indicates whether optional result is required */
    size_t batchSize;                                /*!< Number of batch indices to compute the stochastic gradient.
                                                        If batchSize is equal to the number of terms in objective
                                                        function then no random sampling is performed, and all terms are
                                                        used to calculate the gradient. This parameter is ignored
                                                        if batchIndices is provided. */
};
/* [interface1::Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__INPUT"></a>
 * \brief %Input parameters for the iterative solver algorithm    \DAAL_DEPRECATED
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other);

    ~Input() DAAL_C11_OVERRIDE {}

    /**
     * Returns input NumericTable of the iterative solver algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Returns optional input of the iterative solver algorithm
    * \param[in] id    Identifier of the optional input data
    * \return          %Input data that corresponds to the given identifier
    */
    algorithms::OptionalArgumentPtr get(OptionalInputId id) const;

    /**
    * Returns input NumericTable containing optional data
    * \param[in] id    Identifier of the input numeric table
    * \return          %Input numeric table that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
     * Sets input for the iterative solver algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Sets optional input for the iterative solver algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalInputId id, const algorithms::OptionalArgumentPtr & ptr);

    /**
    * Sets optional input for the algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);
    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULT"></a>
 * \brief Results obtained with the compute() method of the iterative solver algorithm in the batch processing mode    \DAAL_DEPRECATED
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result() : daal::algorithms::Result(lastOptionalResultId + 1) {}

    ~Result() DAAL_C11_OVERRIDE {};

    /**
     * Allocates memory to store the results of the iterative solver algorithm
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Returns result of the iterative solver algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Returns optional result of the iterative solver algorithm
    * \param[in] id    Identifier of the optional result data
    * \return          %Result data that corresponds to the given identifier
    */
    algorithms::OptionalArgumentPtr get(OptionalResultId id) const;

    /**
    * Returns optional result of the algorithm
    * \param[in] id   Identifier of the optional result
    * \return         optional result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
     * Sets the result of the iterative solver algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Sets the optional result of the iterative solver algorithm
    * \param[in] id    Identifier of the result
    * \param[in] ptr   Pointer to the result
    */
    void set(OptionalResultId id, const algorithms::OptionalArgumentPtr & ptr);

    /**
    * Sets optional result of the algorithm
    * \param[in] id    Identifier of the optional result
    * \param[in] ptr   Pointer to the optional result
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the result of the iterative solver algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
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

/** @} */
} // namespace interface1

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__PARAMETER"></a>
 * \brief %Parameter base class for the iterative solver algorithm
 *
 * \snippet optimization_solver/iterative_solver/iterative_solver_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter base class of the iterative solver algorithm
     * \param[in] function_             Objective function represented as sum of functions
     * \param[in] nIterations_          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold_    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] optionalResultReq_    Flag indicating if algorithm-specific result data generation is needed
     * \param[in] batchSize_            Batch size
     */
    Parameter(const sum_of_functions::BatchPtr & function_, size_t nIterations_ = 100, double accuracyThreshold_ = 1.0e-05,
              bool optionalResultReq_ = false, size_t batchSize_ = 1);

    /**
    * Constructs an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter(const Parameter & other);

    /**
    * Copy an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter & operator=(const Parameter & other);

    ~Parameter() DAAL_C11_OVERRIDE {}

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    sum_of_functions::BatchPtr function; /*!< Objective function represented as sum of functions */
    size_t nIterations;                  /*!< Maximal number of iterations of the algorithm */
    double accuracyThreshold;            /*!< Accuracy of the algorithm. The algorithm terminates when
                                                                this accuracy is achieved */
    bool optionalResultRequired;         /*!< Indicates whether optional result is required */
    size_t batchSize;                    /*!< Number of batch indices to compute the stochastic gradient.
                                                        If batchSize is equal to the number of terms in objective
                                                        function then no random sampling is performed, and all terms are
                                                        used to calculate the gradient. This parameter is ignored
                                                        if batchIndices is provided. */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__INPUT"></a>
 * \brief %Input parameters for the iterative solver algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other);

    ~Input() DAAL_C11_OVERRIDE {}

    /**
     * Returns input NumericTable of the iterative solver algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Returns optional input of the iterative solver algorithm
    * \param[in] id    Identifier of the optional input data
    * \return          %Input data that corresponds to the given identifier
    */
    algorithms::OptionalArgumentPtr get(OptionalInputId id) const;

    /**
    * Returns input NumericTable containing optional data
    * \param[in] id    Identifier of the input numeric table
    * \return          %Input numeric table that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
     * Sets input for the iterative solver algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Sets optional input for the iterative solver algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalInputId id, const algorithms::OptionalArgumentPtr & ptr);

    /**
    * Sets optional input for the algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);
    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULT"></a>
 * \brief Results obtained with the compute() method of the iterative solver algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result() : daal::algorithms::Result(lastOptionalResultId + 1) {}

    ~Result() DAAL_C11_OVERRIDE {};

    /**
     * Allocates memory to store the results of the iterative solver algorithm
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Returns result of the iterative solver algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Returns optional result of the iterative solver algorithm
    * \param[in] id    Identifier of the optional result data
    * \return          %Result data that corresponds to the given identifier
    */
    algorithms::OptionalArgumentPtr get(OptionalResultId id) const;

    /**
    * Returns optional result of the algorithm
    * \param[in] id   Identifier of the optional result
    * \return         optional result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
     * Sets the result of the iterative solver algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Sets the optional result of the iterative solver algorithm
    * \param[in] id    Identifier of the result
    * \param[in] ptr   Pointer to the result
    */
    void set(OptionalResultId id, const algorithms::OptionalArgumentPtr & ptr);

    /**
    * Sets optional result of the algorithm
    * \param[in] id    Identifier of the optional result
    * \param[in] ptr   Pointer to the optional result
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the result of the iterative solver algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
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

/** @} */
} // namespace interface2
using interface2::Parameter;
using interface2::Input;
using interface2::Result;
using interface2::ResultPtr;

} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
