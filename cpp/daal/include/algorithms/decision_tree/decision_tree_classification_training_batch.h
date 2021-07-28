/* file: decision_tree_classification_training_batch.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of the interface for Decision tree model-based training in the batch processing mode
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_TRAINING_BATCH_H__
#define __DECISION_TREE_CLASSIFICATION_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/decision_tree/decision_tree_classification_training_types.h"
#include "algorithms/decision_tree/decision_tree_classification_model.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace interface2
{
/**
 * @defgroup decision_tree_classification_training_batch Batch
 * @ingroup decision_tree_classification_training
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for Decision tree model-based training using algorithmFPType precision arithmetic
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for Decision tree model-based training with a specified environment in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~BatchContainer();

    /**
     * Computes the result of Decision tree model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__TRAINING__BATCH"></a>
 * \brief Provides methods for Decision tree model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DT-ALGORITHM">Decision tree algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for Decision tree model-based training, double or float
 * \tparam method           Decision tree training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref decision_tree::classification::interface1::Model "decision_tree::classification::Model" class
 *      - \ref prediction::interface2::Batch "prediction::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef algorithms::decision_tree::classification::training::Input InputType;
    typedef algorithms::decision_tree::classification::Parameter ParameterType;
    typedef algorithms::decision_tree::classification::training::Result ResultType;

    InputType input;         /*!< %Input objects of the algorithm */
    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Batch(size_t nClasses) : parameter(nClasses) { initialize(); }

    /**
     * Constructs a Decision tree training algorithm by copying input objects
     * and parameters of another Decision tree training algorithm in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other) : classifier::training::Batch(other), input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    /**
     * Get input objects for the Decision tree training algorithm
     * \return %Input objects for the Decision tree training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of Decision tree model-based training
     * \return Structure that contains the result of Decision tree model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the results of Decision tree model training algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to a newly allocated Decision tree training algorithm
     * with a copy of the input objects and parameters for this Decision tree training algorithm
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>((classifier::training::InputIface *)(&input), &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in  = &input;
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};

/** @} */
} // namespace interface2

using interface2::BatchContainer;
using interface2::Batch;

} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
