/* file: objective_function_batch.h */
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
//  Implementation of the Objective function types.
//--
*/

#ifndef __OBJECTIVE_FUNCTION_BATCH_H__
#define __OBJECTIVE_FUNCTION_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "objective_function_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{

namespace interface1
{
/** @defgroup objective_function_batch Batch
 * @ingroup objective_function
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__BATCH"></a>
 * \brief Interface for computing the Objective function in the batch processing mode.
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects for the Objective function
 *      - \ref ResultId %Result identifiers for the Objective function
 *
 * \par References
 *      - Input class
 *      - Result class
 */
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::optimization_solver::objective_function::Input     InputType;
    typedef algorithms::optimization_solver::objective_function::Parameter ParameterType;
    typedef algorithms::optimization_solver::objective_function::Result    ResultType;

    /**
     *  Main constructor
     */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs an Objective function by copying input objects and parameters
     * of another Objective function
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other)
    {
        initialize();
    }

    /** Destructor */
    virtual ~Batch() {}

    /**
    * Returns the structure that contains results of the Objective function
    * \return Structure that contains results of the Objective function
    */
    virtual objective_function::ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Sets the memory for storing results of the Objective function
     * \param[in] result  Structure for storing results of the Objective function
     *
     * \return Status of computations
     */
    virtual services::Status setResult(const objective_function::ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult);
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Objective function algorithm with a copy of input objects
     * of this Objective function algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

protected:
    virtual Batch *cloneImpl() const DAAL_C11_OVERRIDE = 0;

    void initialize()
    {
        _result = objective_function::ResultPtr(new ResultType());
    }

protected:
    objective_function::ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
