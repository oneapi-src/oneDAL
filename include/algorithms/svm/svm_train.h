/* file: svm_train.h */
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
//  Implementation of the interface for SVM model-based training in the batch
//  processing mode
//--
*/

#ifndef __SVM_TRAIN_H__
#define __SVM_TRAIN_H__

#include "algorithms/algorithm.h"

#include "algorithms/svm/svm_train_types.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{

namespace interface1
{
/**
 * @defgroup svm_training_batch Batch
 * @ingroup svm_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute results of the SVM training
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 * \tparam method           SVM training computation method, \ref daal::algorithms::svm::training::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for SVM model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of SVM  model-based training in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCH"></a>
 *  \brief %Algorithm class to train the SVM model
 *  \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a>
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 *  \tparam method           SVM training method, \ref Method
 *
 *  \par Enumerations
 *      - \ref classifier::training::InputId Identifiers of SVM training input objects
 *      - \ref classifier::training::ResultId Identifiers of SVM training results
 *      - \ref Method   SVM training methods
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Model "Model" class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = boser>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    /** Default constructor */
    Batch()
    {
        initialize();
    };

    /**
     * Constructs an SVM training algorithm by copying input objects and parameters
     * of another SVM training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : classifier::training::Batch(other)
    {
        initialize();
        parameter = other.parameter;
    }

    virtual ~Batch() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store results of the SVM training algorithm
     * \param[in] result    Structure to store results of the SVM training algorithm
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns structure that contains computed results of the SVM training algorithm
     * \return Structure that contains computed results of the SVM training algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return services::staticPointerCast<Result, classifier::training::Result>(_result);
    }

    /**
     * Resets the training results of the classification algorithm
     */
    void resetResult() DAAL_C11_OVERRIDE
    {
        _result = services::SharedPtr<Result>(new Result());
        _res = NULL;
    }

    /**
     * Returns a pointer to the newly allocated SVM training algorithm with a copy of input objects
     * and parameters of this SVM training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    Parameter parameter;        /*!< Parameters of the algorithm */

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<Result> res = services::staticPointerCast<Result, classifier::training::Result>(_result);
        res->template allocate<algorithmFPType>(&input, _par, (int) method);
        _res = _result.get();
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
