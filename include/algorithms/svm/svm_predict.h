/* file: svm_predict.h */
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
//  Implementation of the interface for SVM model-based prediction
//--
*/

#ifndef __SVM_PREDICT_H__
#define __SVM_PREDICT_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/svm/svm_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * @defgroup svm_prediction_batch Batch
 * @ingroup svm_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the SVM algorithm.
 *        It is associated with the Prediction class
 *        and supports methods to run predictions based on the SVM model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVM prediction algorithm, double or float
 * \tparam method           SVM model-based prediction method, \ref Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for SVM model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of SVM model-based prediction
     *
     * \return Status of computation
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__PREDICTION__BATCH"></a>
 * \brief %Algorithm class for making predictions based on the SVM model
 * <!-- \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a> -->
 *
 * \par Enumerations
 *      - \ref Method                                       %Prediction methods
 *      - \ref classifier::prediction::NumericTableInputId  Input Numeric Table objects
 *                                                          for the SVM prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects
 *                                                          for the SVM prediction algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of prediction results
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref interface1::Result "Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::svm::prediction::Input InputType;
    typedef algorithms::svm::Parameter         ParameterType;
    typedef typename super::ResultType         ResultType;

    InputType input;                /*!< %Input objects of the algorithm */
    ParameterType parameter;        /*!< \ref interface1::Parameter "Parameter" of the algorithm */

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs an SVM prediction algorithm by copying input objects and parameters
     * of another SVM prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : classifier::prediction::Batch(other),
        input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the SVM prediction algorithm
     * \return %Input objects for the SVM prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }


    /**
     * Returns a pointer to the newly allocated SVM prediction algorithm with a copy of input objects
     * and parameters of this SVM prediction algorithm
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
        services::Status s  = _result->allocate<algorithmFPType>(&input, 0, 0);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
