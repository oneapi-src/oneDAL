/* file: sum_of_functions_types.h */
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
//  Implementation of the Sum of functions interface.
//--
*/

#ifndef __SUM_OF_FUNCTIONS_TYPES_H__
#define __SUM_OF_FUNCTIONS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "objective_function_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the Sum of functions
 */
namespace optimization_solver
{
/**
 * @defgroup sum_of_functions Sum of Functions
 * \copydoc daal::algorithms::optimization_solver::sum_of_functions
 * @ingroup objective_function
 * @{
 */
/**
* \brief Contains classes for computing the Sum of functions
*/
namespace sum_of_functions
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__INPUTID"></a>
  * Available identifiers of input objects of the Sum of functions
  */
enum InputId
{
    argument = (int)objective_function::argument, /*!< Numeric table of size 1 x p with input argument of the objective function */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__PARAMETER"></a>
 * \brief %Parameter for the Sum of functions
 *
 * \snippet optimization_solver/objective_function/sum_of_functions_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public objective_function::Parameter
{
    /**
     * Constructs the parameter of Sum of functions
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms,
              data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(),
              const DAAL_UINT64 resultsToCompute = objective_function::gradient);

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other);

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const;

    virtual ~Parameter() {}

    size_t numberOfTerms;                                              /*!< The number of terms in the function */
    data_management::NumericTablePtr batchIndices;   /*!< Numeric table of size 1 x m where m is batch size that represent
                                                                            a batch of indices used to compute the function results, e.g.,
                                                                            value of the sum of the functions. If no indices are provided,
                                                                            all terms will be used in the computations.  */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__INPUT"></a>
 * \brief %Input objects for the Sum of functions
 */
class DAAL_EXPORT Input : public objective_function::Input
{
public:
    /** Default constructor */
    Input(size_t n = 1);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for Sum of functions
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Returns the input numeric table for Sum of functions
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /**
     * Checks the correctness of the numeric table
     * \param[in] nt              Pointer to the numeric table
     * \param[in] argumentName    Name of checked argument
     * \param[in] requiredRows    Number of required rows. If it equal 0 or not mentioned, the numeric table can't have 0 rows
     * \param[in] requiredColumns Number of required columns. If it equal 0 or not mentioned, the numeric table can't have 0 columns
     */
    services::SharedPtr<services::Error> checkTable(data_management::NumericTablePtr nt, const char *argumentName,
                                                    size_t requiredRows = 0, size_t requiredColumns = 0) const;
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;

} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
