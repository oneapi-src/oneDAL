/* file: svm_train.h */
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
     *
     * \return Status of computation
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCH"></a>
 *  \brief %Algorithm class to train the SVM model
 *  <!-- \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a> -->
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
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Model "Model" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = boser>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType         InputType;
    typedef algorithms::svm::Parameter        ParameterType;
    typedef algorithms::svm::training::Result ResultType;

    ParameterType parameter;        /*!< \ref interface1::Parameter "Parameters" of the algorithm */
    InputType input;                /*!< %Input objects of the algorithm */

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
    Batch(const Batch<algorithmFPType, method> &other) : classifier::training::Batch(other),
        parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the SVM training algorithm
     * \return %Input objects for the SVM training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns structure that contains computed results of the SVM training algorithm
     * \return Structure that contains computed results of the SVM training algorithm
     */
    ResultPtr getResult()
    {
        return ResultType::cast(_result);
    }

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
     * Returns a pointer to the newly allocated SVM training algorithm with a copy of input objects
     * and parameters of this SVM training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s  = res->template allocate<algorithmFPType>(&input, _par, (int) method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
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
