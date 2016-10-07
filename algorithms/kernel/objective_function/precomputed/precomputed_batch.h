/* file: precomputed_batch.h */
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
//  Implementation of the precomputed objective function types.
//--
*/

#ifndef __PRECOMPUTED_BATCH_H__
#define __PRECOMPUTED_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_batch.h"
#include "precomputed_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace internal
{
namespace precomputed
{
namespace interface1
{

template<typename algorithmFPType, Method method>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    BatchContainer(daal::services::Environment::env *daalEnv) {}
    virtual ~BatchContainer() {}
    /**
     * Runs implementations of the precomputed objective function in the batch processing mode
     */
    virtual void compute() {/*empty compute*/}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__precomputed__BATCH"></a>
 * \brief Computes the precomputed objective function in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the precomputed objective function, double or float
 * \tparam method           The precomputed objective function computation method
 *
 * \par Enumerations
 *      - Method Computation methods for the precomputed objective function
 *      - InputId  Identifiers of input objects for the precomputed objective function
 *      - ResultId Result identifiers for the precomputed objective function
 *
 * \par References
 *      - <a href="DAAL-REF-PRECOMPUTED-ALGORITHM">The precomputed objective function algorithm description and usage models</a>
 *      - Input class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public sum_of_functions::Batch
{
public:
    Input input;           /*!< %Input data structure */
    Parameter parameter;   /*!< %Parameter data structure */
    /**
     *  Main constructor
     */
    Batch();

    virtual ~Batch() {}

    /**
     * Constructs the precomputed objective function algorithm by copying input objects and parameters
     * of another the precomputed objective function algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) :
        parameter(other.parameter), sum_of_functions::Batch(other.parameter.numberOfTerms, &input, &parameter)
    {
        initialize();
        input.set(sum_of_functions::argument, other.input.get(sum_of_functions::argument));
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    virtual Input *getInput()
    {
        return &input;
    }

    /**
     * Returns a pointer to the newly allocated the precomputed objective function algorithm with a copy of input objects
     * of this the precomputed objective function algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    void allocate()
    {
        allocateResult();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<Result> precomputedResult = services::SharedPtr<Result>(new Result);
        precomputedResult->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _result = precomputedResult;
        _res = _result.get();
    }

    void initialize()
    {
        Analysis<batch>::_ac = new BatchContainer<algorithmFPType, method>(&_env);
        _in  = &input;
        _par = &parameter;
        _result = services::SharedPtr<precomputed::Result>(new precomputed::Result());
    }
};
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace precomputed
} // namespace internal
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
