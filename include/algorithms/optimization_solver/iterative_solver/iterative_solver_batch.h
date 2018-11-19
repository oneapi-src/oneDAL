/* file: iterative_solver_batch.h */
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
//  Implementation of iterative solver interface interface.
//--
*/

#ifndef __ITERATIVE_SOLVER_BATCH_H__
#define __ITERATIVE_SOLVER_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/optimization_solver_batch.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
/**
* \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/** @defgroup iterative_solver_batch Batch
 * @ingroup iterative_solver
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH"></a>
 * \brief Interface for computing the iterative solver in the %batch processing mode.
 */
class DAAL_EXPORT Batch : public  optimization_solver::BatchIface
{
public:
    typedef algorithms::optimization_solver::iterative_solver::Input     InputType;
    typedef algorithms::optimization_solver::iterative_solver::Parameter ParameterType;
    typedef algorithms::optimization_solver::iterative_solver::Result    ResultType;

    Batch()
    {
    }

    /**
     * Constructs a iterative solver algorithm by copying input objects
     * of another iterative solver algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other)
    {
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the iterative solver algorithm
     * \return %Input objects for the iterative solver algorithm
     */
    virtual InputType * getInput() = 0;

    /**
     * Get parameters of the iterative solver algorithm
     * \return Parameters of the iterative solver algorithm
     */
    virtual ParameterType * getParameter() = 0;

    /**
     * Returns the structure that contains results of the iterative solver algorithm
     * \return Structure that contains results of the iterative solver algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Creates user-allocated memory to store results of the iterative solver algorithm
     *
     * \return Status of computations
     */
    virtual services::Status createResult() = 0;

    /**
     * Returns a pointer to the newly allocated iterative solver algorithm with a copy of input objects
     * of this iterative solver algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

protected:
    virtual Batch *cloneImpl() const DAAL_C11_OVERRIDE = 0;

    ResultPtr _result;
};
/** @} */
typedef services::SharedPtr<Batch> BatchPtr;

} // namespace interface1
typedef interface1::Batch Batch;
using interface1::BatchPtr;

} // namespace optimization_solver
} // namespace iterative_solver
} // namespace algorithm
} // namespace daal
#endif
