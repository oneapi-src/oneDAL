/* file: gbt_regression_init_types.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the interface for initializing gradient boosted trees
//  regression training algorithm in the distributed processing mode
//--
*/

#ifndef __GBT_REGRESSION_TRAINING_INIT_TYPES_H__
#define __GBT_REGRESSION_TRAINING_INIT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/gradient_boosted_trees/gbt_training_parameter.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
/**
 * @defgroup gbt_regression_training_init Training
 * \copydoc daal::algorithms::gbt::regression::training
 * @ingroup gbt_regression_training_init
 * @{
 */
/**
 * \brief
 */
namespace init
{
    /**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__METHOD"></a>
 * \brief Computation methods for gradient boosted trees regression model-based training
 */
enum Method
{
    defaultDense = 0  /*!< Default training method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__STEP1LOCALNUMERICTABLEINPUTID"></a>
 *
 */
enum Step1LocalNumericTableInputId
{
    step1LocalData,
    step1LocalDependentVariables,
    lastStep1LocalInputId = step1LocalDependentVariables
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 *
 */
enum DistributedPartialResultStep1Id
{
    step1BinBorders,
    step1BinSizes,
    step1MeanDependentVariable,
    step1NumberOfRows,
    lastDistributedPartialResultStep1Id = step1NumberOfRows
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__STEP2MASTERCOLLECTIONINPUTID"></a>
 *
 */
enum Step2MasterCollectionInputId
{
    step2MeanDependentVariable,
    step2NumberOfRows,
    step2BinBorders,
    step2BinSizes,
    lastStep2MasterCollectionInputId = step2BinSizes
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP2NUMERICTABLEID"></a>
 *
 */
enum DistributedPartialResultStep2NumericTableId
{
    step2MergedBinBorders,
    step2BinQuantities,
    step2InitialResponse,
    lastDistributedPartialResultStep2NumericTableId = step2InitialResponse
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP2COLLECTIONID"></a>
 *
 */
enum DistributedPartialResultStep2CollectionId
{
    step2BinValues = lastDistributedPartialResultStep2NumericTableId + 1,
    lastDistributedPartialResultStep2CollectionId = step2BinValues
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__STEP3LOCALNUMERICTABLEINPUTID"></a>
 *
 */
enum Step3LocalNumericTableInputId
{
    step3MergedBinBorders,
    step3BinQuantities,
    step3LocalData,
    step3InitialResponse,
    lastStep3LocalInputId = step3InitialResponse
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 *
 */
enum DistributedPartialResultStep3Id
{
    step3BinnedData,
    step3TransposedBinnedData,
    step3Response,
    step3TreeOrder,
    lastDistributedPartialResultStep3Id = step3TreeOrder
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__PARAMETER"></a>
 * \brief
 *
 * \snippet
 */
class DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
public:
    /**
     *  Parameter constructor
     *  \param[in] _nClasses       Number of classes
     *  \param[in] _maxBins        Maximal number of discrete bins to bucket continuous features
     *  \param[in] _minBinSize     Minimal number of observations in a bin
     */
    Parameter(size_t _maxBins = 256, size_t _minBinSize = 5);

    /**
     * Constructs parameters of the algorithm that computes ... for ... algorithm
     * by copying another parameters object
     * \param[in] other    Parameters of the K-Means algorithm
     */
    Parameter(const Parameter &other);

    virtual ~Parameter() {}

    services::Status check() const DAAL_C11_OVERRIDE;

    size_t maxBins;    /*!< Used with 'inexact' split finding method only.
                            Maximal number of discrete bins to bucket continuous features.
                            Default is 256. Increasing the number results in higher computation costs */
    size_t minBinSize; /*!< Used with 'inexact' split finding method only.
                            Minimal number of observations in a bin. Default is 5 */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for model-based training in the distributed processing mode
 */
template<ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief %Input objects for model-based training in the first step of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step2Master> : public daal::algorithms::Input
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
    data_management::DataCollectionPtr get(Step2MasterCollectionInputId id) const;

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step2MasterCollectionInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step2MasterCollectionInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief
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
    data_management::NumericTablePtr get(DistributedPartialResultStep2NumericTableId id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2NumericTableId id, const data_management::NumericTablePtr &ptr);

    /**
     * Returns a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep2CollectionId id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2CollectionId id, const data_management::DataCollectionPtr &ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDINPUT_STEP3LOCAL"></a>
 * \brief %Input objects for model-based training in the first step of the distributed processing mode
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
     * Sets an input object for the model-based training
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalNumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the parameters and input objects for the model-based training
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief
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
    data_management::NumericTablePtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Sets a partial result of the model-based training
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep3Id id, const data_management::NumericTablePtr &ptr);

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
} // namespace interface1
using interface1::Parameter;
using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep1Ptr;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep2Ptr;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep3Ptr;

} // namespace init
/** @} */
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif // __GBT_REGRESSION_TRAINING_INIT_TYPES_H__
