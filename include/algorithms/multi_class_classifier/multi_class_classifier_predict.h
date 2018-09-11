/* file: multi_class_classifier_predict.h */
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
//  Implementation of the interface for multi-class classifier model-based prediction
//  in the batch processing mode
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_PREDICT_H__
#define __MULTI_CLASS_CLASSIFIER_PREDICT_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_predict_types.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_train_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * \brief Contains classes for prediction based on multi-class classifier models
 */
namespace prediction
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * @defgroup multi_class_classifier_prediction_batch Batch
 * @ingroup multi_class_classifier_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the  multi-class classifier prediction algorithm
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the multi-class classifier algorithm, double or float
 * \tparam pmethod          Computation method for the algorithm, \ref prediction::Method
 * \tparam tmethod          Computation method that was used to train the multi-class classifier model, \ref training::Method
 */
template<typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for multi-class classifier model-based prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of multi-class classifier model-based prediction in the batch processing mode
     *
     * \return Status of computation
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__BATCH"></a>
 *  \brief Provides methods to run implementations of the multi-class classifier prediction algorithm
 *  <!-- \n<a href="DAAL-REF-MULTICLASSCLASSIFIER-ALGORITHM">Multi-class classifier algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for multi-class classifier prediction algorithm, double or float
 *  \tparam pmethod          Computation method for the algorithm, \ref prediction::Method
 *  \tparam tmethod          Computation method that was used to train the multi-class classifier model, \ref training::Method
 *
 *  \par Enumerations
 *      - \ref Method Computation methods for the multi-class classifier prediction algorithm
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input NumericTable objects
 *                                                          for the multi-class classifier prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects
 *                                                          for the multi-class classifier prediction algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of the results of the multi-class classifier prediction algorithm
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref classifier::prediction::interface1::Result "classifier::prediction::Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, prediction::Method pmethod = defaultDense, training::Method tmethod = training::oneAgainstOne>
class Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::multi_class_classifier::prediction::Input InputType;
    typedef algorithms::multi_class_classifier::Parameter         ParameterType;
    typedef typename super::ResultType                            ResultType;

    InputType input;                /*!< Input objects of the algorithm */
    ParameterType parameter;        /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /**
     * Default constructor
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED Batch() : parameter(0)
    {
        initialize();
    }

    /**
     * Default constructor
     * \param[in] nClasses                         Number of classes
     */
    Batch(size_t nClasses) : parameter(nClasses)
    {
        initialize();
    }

    /**
     * Constructs multi-class classifier prediction algorithm by copying input objects and parameters
     * of another multi-class classifier prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, pmethod, tmethod> &other) : classifier::prediction::Batch(other),
        parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the multi-class classifier prediction algorithm
     * \return %Input objects for the multi-class classifier prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)pmethod; }

    /**
     * Returns a pointer to the newly allocated multi-class classifier prediction algorithm
     * with a copy of input objects and parameters of this multi-class classifier prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, pmethod, tmethod> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, pmethod, tmethod> >(cloneImpl());
    }

protected:

    virtual Batch<algorithmFPType, pmethod, tmethod> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, pmethod, tmethod>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s  = _result->allocate<algorithmFPType>(&input, &parameter, (int) pmethod);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, pmethod, tmethod)(&_env);
        _par = &parameter;
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
#endif
