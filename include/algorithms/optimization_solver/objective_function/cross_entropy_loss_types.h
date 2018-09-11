/* file: cross_entropy_loss_types.h */
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
//  Implementation of cross-entropy loss objective function
//--
*/

#ifndef __CROSS_ENTROPY_LOSS_TYPES_H__
#define __CROSS_ENTROPY_LOSS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the Cross-entropy loss objective function
 */
namespace optimization_solver
{
/**
 * @defgroup cross_entropy_loss Cross-entropy loss Algorithm
 * \copydoc daal::algorithms::optimization_solver::cross_entropy_loss
 * @ingroup objective_function
 * @{
 */
/**
* \brief Contains classes for computing the Cross-entropy loss objective function
*/
namespace cross_entropy_loss
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY_LOSS__INPUTID"></a>
  * Available identifiers of input objects of the Cross-entropy loss objective function
  */
enum InputId
{
    argument = (int)sum_of_functions::argument, /*!< Numeric table of size 1 x p with input argument of the objective function */
    data,                                       /*!< Numeric table of size n x p with data */
    dependentVariables,                         /*!< Numeric table of size n x 1 with dependent variables */
    lastInputId = dependentVariables
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY__METHOD"></a>
 * Available methods for computing results of Cross-entropy loss objective function
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
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY__PARAMETER"></a>
 * \brief %Parameter for Cross-entropy loss objective function
 *
 * \snippet optimization_solver/objective_function/cross_entropy_loss_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public sum_of_functions::Parameter
{
    /**
     * Constructs the parameter of Cross-entropy loss objective function
     * \param[in] nClasses         The number of different values of dependent variable
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t nClasses, size_t numberOfTerms,
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

    bool interceptFlag;  /*!< Whether the intercept needs to be computed. Default is true */
    float penaltyL1;     /*!< L1 regularization coefficient. Default is 0 (not applied) */
    float penaltyL2;     /*!< L2 regularization coefficient. Default is 0 (not applied) */
    size_t nClasses;     /*!< Number of classes (different values of dependent variable) */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY__INPUT"></a>
 * \brief %Input objects for the Cross-entropy loss objective function
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
     * Sets one input object for Cross-entropy loss objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Returns the input numeric table for Cross-entropy loss objective function
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

} // namespace cross_entropy_loss
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
