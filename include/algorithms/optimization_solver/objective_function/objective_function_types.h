/* file: objective_function_types.h */
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
//  Implementation of the Objective function interface.
//--
*/

#ifndef __OBJECTIVE_FUNCTION_TYPES_H__
#define __OBJECTIVE_FUNCTION_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the Objective function
 */
namespace optimization_solver
{
/**
 * @defgroup objective_function Objective Function
 * \copydoc daal::algorithms::objective_function
 * @ingroup optimization_solver
 * @{
 */
/**
* \brief Contains classes for computing the Objective function
*/
namespace objective_function
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__INPUTID"></a>
  * Available identifiers of input objects of the Objective function
  */
enum InputId
{
    argument,  /*!< Numeric table of size 1 x p with input argument of the objective function */
    lastInputId = argument
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the Objective function result
 */
enum ResultToComputeId
{
    gradient = 0x00000001ULL, /*!< Numeric table of size 1 x p with the gradient of the objective function in the given argument */
    value    = 0x00000002ULL, /*!< Numeric table of size 1 x 1 with the value    of the objective function in the given argument */
    hessian  = 0x00000004ULL  /*!< Numeric table of size p x p with the hessian  of the objective function in the given argument */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTID"></a>
 * Available identifiers of results of the Objective function
 */
enum ResultId
{
    gradientIdx, /*!< Index of the gradient numeric table in the result collection */
    valueIdx,    /*!< Index of the value numeric table in the result collection */
    hessianIdx,  /*!< Index of the hessian numeric table in the result collection */
    lastResultId = hessianIdx
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__PARAMETER"></a>
 * \brief %Parameter for the Objective function
 *
 * \snippet optimization_solver/objective_function/objective_function_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter of the Objective function
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(const DAAL_UINT64 resultsToCompute = gradient);

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other);

    virtual ~Parameter() {}

    DAAL_UINT64 resultsToCompute;  /*!< 64 bit integer flag that indicates the results to compute */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__INPUT"></a>
 * \brief %Input objects for the Objective function
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input(size_t n = 1);

    /** Copy constructor */
    Input(const Input& other);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for Objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Returns the input numeric table for Objective function
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the Objective function in the batch processing mode
 */
class DAAL_EXPORT Result: public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /** Default constructor */
    Result();

    /** Destructor */
    virtual ~Result() {};

    /**
     * Allocates memory for storing results of the Objective function
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Sets the result of the Objective function
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the numeric table with the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Returns the collection of the results of the Objective function
     * \param[in] id   Identifier of the result
     * \return         %Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Checks the result of the Objective function
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method
    *
     * \return Status of computations
    */
    virtual services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
