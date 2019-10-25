/* file: gbt_regression_training_types.h */
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
//  Implementation of the gradient boosted trees regression training algorithm interface
//--
*/

#ifndef __GBT_REGRESSION_TRAINIG_TYPES_H__
#define __GBT_REGRESSION_TRAINIG_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "algorithms/gradient_boosted_trees/gbt_training_parameter.h"
#include "algorithms/regression/regression_training_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the gradient boosted trees algorithm
 */
namespace gbt
{
namespace regression
{
/**
 * @defgroup gbt_regression_training Training
 * \copydoc daal::algorithms::gbt::regression::training
 * @ingroup gbt_regression
 * @{
 */
/**
 * \brief Contains a class for model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__METHOD"></a>
 * \brief Computation methods for gradient boosted trees classification model-based training
 */
enum Method
{
    xboost       = 0, /*!< Extreme boosting (second-order approximation of objective function,
                           regularization on number of leaves and their weights), Chen et al. */
    defaultDense = 0  /*!< Default training method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__LOSS_FUNCTION_TYPE"></a>
* \brief Loss function type
*/
enum LossFunctionType
{
    squared, /* L(y,f) = ([y-f(x)]^2)/2 */
    custom   /* Should be differentiable up to the second order */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__INPUTID"></a>
 * \brief Available identifiers of input objects for model-based training
 */
enum InputId
{
    data              = algorithms::regression::training::data,               /*!< %Input data table */
    dependentVariable = algorithms::regression::training::dependentVariables, /*!< %Values of the dependent variable for the input data */
    lastInputId       = dependentVariable
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the result of model-based training
 */
enum ResultId
{
    model        = algorithms::regression::training::model, /*!< model */
    lastResultId = model
};

enum ResultNumericTableId
{
    variableImportanceByWeight = lastResultId + 1,
    variableImportanceByTotalCover,
    variableImportanceByCover,
    variableImportanceByTotalGain,
    variableImportanceByGain,
    lastResultNumericTableId = variableImportanceByGain
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP1LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step1LocalNumericTableInputId
{
    step1BinnedData,                                           /*!<  */
    step1DependentVariable,                                    /*!<  */
    step1InputResponse,                                        /*!<  */
    step1InputTreeStructure,                                   /*!<  */
    step1InputTreeOrder,                                       /*!<  */
    lastStep1LocalNumericTableInputId = step1InputTreeOrder
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * Available types of partial results of model-based training in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep1Id
{
    response,                                               /*!<  */
    optCoeffs,                                              /*!<  */
    treeOrder,                                              /*!<  */
    finalizedTree,                                          /*!<  */
    step1TreeStructure,                                     /*!<  */
    lastDistributedPartialResultStep1Id = step1TreeStructure
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP2LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step2LocalNumericTableInputId
{
    step2InputTreeStructure,                                   /*!<  */
    lastStep2LocalNumericTableInputId = step2InputTreeStructure
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Available types of partial results of model-based training in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    finishedFlag,                                       /*!<  */
    lastDistributedPartialResultStep2Id = finishedFlag
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP3LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step3LocalNumericTableInputId
{
    step3BinnedData,                                           /*!<  */
    step3BinSizes,                                             /*!<  */
    step3InputTreeStructure,                                   /*!<  */
    step3InputTreeOrder,                                       /*!<  */
    step3OptCoeffs,                                            /*!<  */
    lastStep3LocalNumericTableInputId = step3OptCoeffs
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP3LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step3LocalCollectionInputId
{
    step3ParentHistograms = lastStep3LocalNumericTableInputId + 1,                                           /*!<  */
    lastStep3LocalCollectionInputId = step3ParentHistograms
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Available types of partial results of model-based training in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    histograms,                                       /*!<  */
    lastDistributedPartialResultStep3Id = histograms
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP4LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step4LocalNumericTableInputId
{
    step4InputTreeStructure,                                     /*!<  */
    lastStep4LocalNumericTableInputId = step4InputTreeStructure
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP4LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step4LocalCollectionInputId
{
    step4FeatureIndices = lastStep4LocalNumericTableInputId + 1,          /*!<  */
    step4ParentTotalHistograms,                                           /*!<  */
    step4PartialHistograms,                                               /*!<  */
    lastStep4LocalCollectionInputId = step4PartialHistograms
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4ID"></a>
 * Available types of partial results of model-based training in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep4Id
{
    totalHistograms,                                       /*!<  */
    bestSplits,                                            /*!<  */
    lastDistributedPartialResultStep4Id = bestSplits
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP5LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step5LocalNumericTableInputId
{
    step5BinnedData,                                             /*!<  */
    step5TransposedBinnedData,                                   /*!<  */
    step5BinSizes,                                               /*!<  */
    step5InputTreeStructure,                                     /*!<  */
    step5InputTreeOrder,                                         /*!<  */
    lastStep5LocalNumericTableInputId = step5InputTreeOrder
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP5LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step5LocalCollectionInputId
{
    step5PartialBestSplits = lastStep5LocalNumericTableInputId + 1,                                           /*!<  */
    lastStep5LocalCollectionInputId = step5PartialBestSplits
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP5ID"></a>
 * Available types of partial results of model-based training in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep5Id
{
    step5TreeStructure,                                        /*!<  */
    step5TreeOrder,                                            /*!<  */
    lastDistributedPartialResultStep5Id = step5TreeOrder
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP6LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step6LocalNumericTableInputId
{
    step6InitialResponse,                                             /*!<  */
    lastStep6LocalNumericTableInputId = step6InitialResponse
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP6LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for model-based training in the first step
 * of the distributed processing mode
 */
enum Step6LocalCollectionInputId
{
    step6BinValues = lastStep6LocalNumericTableInputId + 1,                /*!<  */
    step6FinalizedTrees,                                                   /*!<  */
    lastStep6LocalCollectionInputId = step6FinalizedTrees
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP6ID"></a>
 * Available types of partial results of model-based training in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep6Id
{
    partialModel,                                                /*!<  */
    lastDistributedPartialResultStep6Id = partialModel
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__PARAMETER"></a>
 * \brief Parameters for the gradient boosted trees algorithm
 *
 * \snippet gradient_boosted_trees/gbt_regression_training_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter : public daal::algorithms::Parameter, public daal::algorithms::gbt::training::Parameter
{
public:
    Parameter();
    services::Status check() const DAAL_C11_OVERRIDE;

    LossFunctionType loss;     /*!< Loss function type */
    DAAL_UINT64 varImportance; /*!< 64 bit integer flag VariableImportanceModes that indicates the variable importance computation modes */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__INPUT"></a>
 * \brief %Input objects for model-based training
 */
class DAAL_EXPORT Input : public algorithms::regression::training::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other) : algorithms::regression::training::Input(other) {}

    virtual ~Input() {};

    /**
     * Returns an input object for model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for model-based training
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
    * Checks an input object for the gradient boosted trees algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of model-based training
 */
class DAAL_EXPORT Result : public algorithms::regression::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Allocates memory to store the result of model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of model-based training
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method);

    /**
     * Returns the result of model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    gbt::regression::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const ModelPtr & value);

    /**
     * Returns the result of model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
     * Sets the result of model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr & value);

    /**
     * Checks the result of model-based training
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for model-based training in the distributed processing mode
 */
template<ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for model-based training in the first step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step1LocalNumericTableInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step1LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of model-based training in the first step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep1 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep1);
    /** Default constructor */
    DistributedPartialResultStep1();

    virtual ~DistributedPartialResultStep1() {}

    /**
     * Returns a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep1Id id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep1Id id, const data_management::NumericTablePtr &ptr);

    /**
     * Allocates memory to store a partial result of the model-based training
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks a partial result of the model-based training
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep1> DistributedPartialResultStep1Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT_STEP2LOCAL"></a>
 * \brief %Input objects for model-based training in the second step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step2Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step2LocalNumericTableInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step2LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of model-based training in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep2 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep2);
    /** Default constructor */
    DistributedPartialResultStep2();

    virtual ~DistributedPartialResultStep2() {}

    /**
     * Returns a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2Id id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr &ptr);

    /**
     * Allocates memory to store a partial result of the model-based training
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks a partial result of the model-based training
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep2> DistributedPartialResultStep2Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT_STEP3LOCAL"></a>
 * \brief %Input objects for model-based training in the third step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step3Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step3LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step3LocalCollectionInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalCollectionInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step3LocalCollectionInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of model-based training in the third step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep3);
    /** Default constructor */
    DistributedPartialResultStep3();

    virtual ~DistributedPartialResultStep3() {}

    /**
     * Returns a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep3Id id, const data_management::DataCollectionPtr &ptr);

    /**
     * Allocates memory to store a partial result of the model-based training
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks a partial result of the model-based training
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep3> DistributedPartialResultStep3Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT_STEP4LOCAL"></a>
 * \brief %Input objects for model-based training in the fourth step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step4Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step4LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step4LocalCollectionInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalCollectionInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of model-based training in the fourth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep4 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep4);
    /** Default constructor */
    DistributedPartialResultStep4();

    virtual ~DistributedPartialResultStep4() {}

    /**
     * Returns a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep4Id id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep4Id id, const data_management::DataCollectionPtr &ptr);

    /**
     * Allocates memory to store a partial result of the model-based training
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks a partial result of the model-based training
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep4> DistributedPartialResultStep4Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT_STEP5LOCAL"></a>
 * \brief %Input objects for model-based training in the fifth step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step5Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step5LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step5LocalCollectionInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step5LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step5LocalCollectionInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step5LocalCollectionInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP5"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of model-based training in the fifth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep5 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep5);
    /** Default constructor */
    DistributedPartialResultStep5();

    virtual ~DistributedPartialResultStep5() {}

    /**
     * Returns a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep5Id id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep5Id id, const data_management::NumericTablePtr &ptr);

    /**
     * Allocates memory to store a partial result of the model-based training
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks a partial result of the model-based training
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep5> DistributedPartialResultStep5Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDINPUT_STEP6LOCAL"></a>
 * \brief %Input objects for model-based training in the sixth step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step6Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step6LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step6LocalCollectionInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step6LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step6LocalCollectionInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step6LocalCollectionInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP6"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of model-based training in the sixth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep6 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep6);
    /** Default constructor */
    DistributedPartialResultStep6();

    virtual ~DistributedPartialResultStep6() {}

    /**
     * Returns the partial result of model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Result that corresponds to the given identifier
     */
    gbt::regression::ModelPtr get(DistributedPartialResultStep6Id id) const;

    /**
     * Sets the partial result of model-based training
     * \param[in] id      Identifier of the partial result
     * \param[in] value   Result
     */
    void set(DistributedPartialResultStep6Id id, const ModelPtr &value);

    /**
     * Allocates memory to store a partial result of the model-based training
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks a partial result of the model-based training
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep6> DistributedPartialResultStep6Ptr;

} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep1Ptr;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep2Ptr;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep3Ptr;
using interface1::DistributedPartialResultStep4;
using interface1::DistributedPartialResultStep4Ptr;
using interface1::DistributedPartialResultStep5;
using interface1::DistributedPartialResultStep5Ptr;
using interface1::DistributedPartialResultStep6;
using interface1::DistributedPartialResultStep6Ptr;

} // namespace training
/** @} */
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
