/* file: iterative_solver_batch.h */
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
class DAAL_EXPORT Batch : public  optimization_solver::Batch
{
public:
    Input *input;           /*!< %Input of the algorithm */
    Parameter *parameter;   /*!< %Parameters of the algorithm */

    Batch() : input(NULL), parameter(NULL)
    {
    }

    /**
     * Constructs a iterative solver algorithm by copying input objects
     * of another iterative solver algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other) :input(other.input), parameter(other.parameter)
    {
    }

    virtual ~Batch() {}

    /**
    * Returns the structure that contains results of the iterative solver algorithm
    * \return Structure that contains results of the iterative solver algorithm
    */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Creates user-allocated memory to store results of the iterative solver algorithm
     */
    virtual void createResult() = 0;

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

    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace optimization_solver
} // namespace iterative_solver
} // namespace algorithm
} // namespace daal
#endif
