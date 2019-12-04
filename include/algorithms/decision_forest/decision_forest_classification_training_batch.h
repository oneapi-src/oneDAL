/* file: decision_forest_classification_training_batch.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the interface for Decision forest model-based training
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_TRAINING_BATCH_H__
#define __DECISION_FOREST_CLASSIFICATION_TRAINING_BATCH_H__

#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/decision_forest/decision_forest_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace interface1
{
/**
 * @defgroup decision_forest_classification_training_batch Batch
 * @ingroup decision_forest_classification_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of Decision forest model-based training.
 *        This class is associated with daal::algorithms::decision_forest::classification::training::Batch class  \DAAL_DEPRECATED
*
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           Decision forest model training method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for Decision forest model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    DAAL_DEPRECATED ~BatchContainer();
    /**
     * Computes the result of Decision forest model-based training in the batch processing mode
     * \return Status of computations
     */
    DAAL_DEPRECATED services::Status compute() DAAL_C11_OVERRIDE;
    DAAL_DEPRECATED services::Status setupCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__BATCH"></a>
 * \brief Trains model of the Decision forest algorithms in the batch processing mode   \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-DECISION_FOREST__CLASSIFICATION-ALGORITHM">Decision forest algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for Decision forest, double or float
 * \tparam method           Decision forest computation method, \ref daal::algorithms::decision_forest::classification::training::Method
 *
 * \par Enumerations
 *      - \ref Method                         Decision forest training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the Decision forest training algorithm
 *      - \ref classifier::training::ResultId Identifiers of Decision forest training results
 *
 * \par References
 *      - \ref decision_forest::classification::interface1::Model "Model" class
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::interface1::Batch
{
public:
    typedef classifier::training::interface1::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::decision_forest::classification::training::interface1::Parameter ParameterType;
    typedef algorithms::decision_forest::classification::training::Result ResultType;

    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */
    InputType input;         /*!< %Input data structure */

    /**
     * Constructs the Decision forest training algorithm
     * \param[in] nClasses  Number of classes
     */
    DAAL_DEPRECATED Batch(size_t nClasses) : parameter(nClasses)
    {
        initialize();
        parameter.minObservationsInLeafNode = 1;
    }

    /**
     * Constructs a Decision forest training algorithm by copying input objects and parameters
     * of another Decision forest training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    DAAL_DEPRECATED Batch(const Batch<algorithmFPType, method> & other)
        : classifier::training::interface1::Batch(other), parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    DAAL_DEPRECATED_VIRTUAL virtual ~Batch() {}

    /**
     * Get input objects for the Decision forest training algorithm
     * \return %Input objects for the Decision forest training algorithm
     */
    DAAL_DEPRECATED InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    DAAL_DEPRECATED_VIRTUAL virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of Decision forest training
     * \return Structure that contains results of Decision forest training
     */
    DAAL_DEPRECATED ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    DAAL_DEPRECATED services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Decision forest training algorithm with a copy of input objects
     * and parameters of this Decision forest training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    DAAL_DEPRECATED services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    DAAL_DEPRECATED_VIRTUAL virtual services::Status checkComputeParams() DAAL_C11_OVERRIDE;

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface1

namespace interface2
{
/**
 * @defgroup decision_forest_classification_training_batch Batch
 * @ingroup decision_forest_classification_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of Decision forest model-based training.
 *        This class is associated with daal::algorithms::decision_forest::classification::training::Batch class
*
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           Decision forest model training method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for Decision forest model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of Decision forest model-based training in the batch processing mode
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    services::Status setupCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__BATCH"></a>
 * \brief Trains model of the Decision forest algorithms in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DECISION_FOREST__CLASSIFICATION-ALGORITHM">Decision forest algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for Decision forest, double or float
 * \tparam method           Decision forest computation method, \ref daal::algorithms::decision_forest::classification::training::Method
 *
 * \par Enumerations
 *      - \ref Method                         Decision forest training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the Decision forest training algorithm
 *      - \ref classifier::training::ResultId Identifiers of Decision forest training results
 *
 * \par References
 *      - \ref decision_forest::classification::interface1::Model "Model" class
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::decision_forest::classification::training::Parameter ParameterType;
    typedef algorithms::decision_forest::classification::training::Result ResultType;

    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */
    InputType input;         /*!< %Input data structure */

    /**
     * Constructs the Decision forest training algorithm
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses) : parameter(nClasses)
    {
        initialize();
        parameter.minObservationsInLeafNode = 1;
    }

    /**
     * Constructs a Decision forest training algorithm by copying input objects and parameters
     * of another Decision forest training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other) : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the Decision forest training algorithm
     * \return %Input objects for the Decision forest training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of Decision forest training
     * \return Structure that contains results of Decision forest training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Decision forest training algorithm with a copy of input objects
     * and parameters of this Decision forest training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    virtual services::Status checkComputeParams() DAAL_C11_OVERRIDE;

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface2
using interface2::BatchContainer;
using interface2::Batch;

} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif // __LOGIT_BOOST_TRAINING_BATCH_H__
