/* file: mse_types.h */
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
//  Implementation of Mean squared error objective function interface.
//--
*/

#ifndef __MSE_TYPES_H__
#define __MSE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the Mean squared error objective function
 */
namespace optimization_solver
{
/**
 * @defgroup mse Mean Squared Error Algorithm
 * \copydoc daal::algorithms::optimization_solver::mse
 * @ingroup objective_function
 * @{
 */
/**
* \brief Contains classes for computing the Mean squared error objective function
*/
namespace mse
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUTID"></a>
  * Available identifiers of input objects of the Mean squared error objective function
  */
enum InputId
{
    argument = (int)sum_of_functions::argument, /*!< Numeric table of size 1 x p with input argument of the objective function */
    data,                                       /*!< Numeric table of size n x p with data */
    dependentVariables,                         /*!< Numeric table of size n x 1 with dependent variables */
    lastInputId = dependentVariables
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__METHOD"></a>
 * Available methods for computing results of Mean squared error objective function
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__PARAMETER"></a>
 * \brief %Parameter for Mean squared error objective function
 *
 * \snippet optimization_solver/objective_function/mse_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public sum_of_functions::Parameter
{
    /**
     * Constructs the parameter of Mean squared error objective function
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
     *
     * \return Status of computations
     */
    virtual services::Status check() const;

    virtual ~Parameter() {}
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUT"></a>
 * \brief %Input objects for the Mean squared error objective function
 */
class DAAL_EXPORT Input : public sum_of_functions::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input& other);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for Mean squared error objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Returns the input numeric table for Mean squared error objective function
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;

} // namespace mse
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
