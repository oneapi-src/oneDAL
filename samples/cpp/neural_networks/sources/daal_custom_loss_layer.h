/* file: daal_custom_loss_layer.h */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
!  Content:
!    Interface and implementation of user-defined neural network loss layer
!    algorithm in Intel DAAL style
!
!******************************************************************************/

#include "daal.h"

using namespace daal;
using namespace daal::algorithms::neural_networks;

/* namespace for the new neural network layer algorithm */
namespace new_loss_layer
{
/**
 * \brief Available identifiers of input objects for the custom loss layer
 */
enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1, /*!< Data processed at the forward stage of the layer */
    auxGroundTruth,                             /*!< Tensor that stores ground truth data for the forward custom loss layer */
    lastLayerDataId = auxGroundTruth
};

/**
 * \brief Parameters for the custom loss layer
 */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the custom loss layer
     */
    Parameter() {}

    /**
     * Checks the correctness of the parameter
     */
    services::Status check() const
    {
        return services::Status();
    }
};

/**
 * \brief Input objects for the forward custom loss layer
 */
class DAAL_EXPORT ForwardInput : public layers::loss::forward::Input
{
public:
    typedef layers::loss::forward::Input super;
    /** Default constructor */
    ForwardInput() {}

    /** Copy constructor */
    ForwardInput(const ForwardInput& other) : super(other) {}

    /**
     * Returns an input object for the forward custom loss layer
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    using layers::loss::forward::Input::get;

    /**
     * Sets an input object for the forward custom loss layer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    using layers::loss::forward::Input::set;

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
     * Checks an input object for the layer algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        services::Status s;
        const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
        if(!parameter->predictionStage)
        {
            if(Argument::size() != 5) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

            TensorPtr dataTensor = get(layers::forward::data);
            TensorPtr groundTruthTensor = get(layers::loss::forward::groundTruth);

            DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), "data"));
            const services::Collection<size_t> &inputDims = dataTensor->getDimensions();

            DAAL_CHECK_STATUS(s, checkTensor(groundTruthTensor.get(), "groundTruth"));
            const services::Collection<size_t> &gtDims = groundTruthTensor->getDimensions();

            DAAL_CHECK_EX(dataTensor->getSize() == groundTruthTensor->getSize(), services::ErrorIncorrectSizeOfDimensionInTensor, services::ParameterName, "groundTruth");
            DAAL_CHECK_EX(gtDims.size() == 1 || gtDims.size() == inputDims.size() , services::ErrorIncorrectNumberOfDimensionsInTensor, services::ParameterName, "data");
            DAAL_CHECK_EX(gtDims[0] == inputDims[0] , services::ErrorIncorrectSizeOfDimensionInTensor, services::ParameterName, "data");
        }
        else
        {
            if(Argument::size() != 5) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

            TensorPtr dataTensor = get(layers::forward::data);

            DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), "data"));
        }
        return s;
    }

    virtual ~ForwardInput() {}
};

/**
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward custom loss layer
 */
class DAAL_EXPORT ForwardResult : public layers::loss::forward::Result
{
public:
    /** Default constructor */
    ForwardResult() : layers::loss::forward::Result() {}
    virtual ~ForwardResult() {}

    /**
     * Returns result of the forward custom loss layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    using layers::loss::forward::Result::get;

    /**
     * Sets the result of the forward custom loss layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    using layers::loss::forward::Result::set;

    /**
     * Returns result of the forward custom loss layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const
    {
        layers::LayerDataPtr layerData = layers::LayerData::cast<SerializationIface>(Argument::get(layers::forward::resultForBackward));
        if(!layerData)
            return data_management::TensorPtr();
        return Tensor::cast<SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of the forward custom loss layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(LayerDataId id, const data_management::TensorPtr &value)
    {
        layers::LayerDataPtr layerData = layers::LayerData::cast<SerializationIface>(Argument::get(layers::forward::resultForBackward));
        if (!layerData) return;
        (*layerData)[id] = value;
    }

    /**
     * Checks the result of the forward custom loss layer
     * \param[in] input   Input object for the algorithm
     * \param[in] par     Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        services::Status s;
        const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
        if(!parameter->predictionStage)
        {
            const ForwardInput *in = static_cast<const ForwardInput * >(input);
            DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, par, method));

            services::Collection<size_t> valueDim(1);
            valueDim[0] = 1;
            DAAL_CHECK_STATUS(s, checkTensor(get(layers::forward::value).get(), "value", &valueDim));
            DAAL_CHECK_STATUS(s, checkTensor(get(auxData).get(), "auxData", &(in->get(layers::forward::data)->getDimensions())));
            DAAL_CHECK_STATUS(s, checkTensor(get(auxGroundTruth).get(), "auxGroundTruth", &(in->get(layers::loss::forward::groundTruth)->getDimensions())));
        }
        else
        {
            const ForwardInput *in = static_cast<const ForwardInput * >(input);
            services::Status s;
            DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, par, method));

            TensorPtr dataTensor = in->get(layers::forward::data);
            const services::Collection<size_t> &inputDims = dataTensor->getDimensions();

            DAAL_CHECK_STATUS(s, checkTensor(get(layers::forward::value).get(), "value", &(inputDims)));
        }
        return s;
    }

    /**
     * Allocates memory to store the result of the forward custom loss layer
     * \param[in] input     Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        services::Status s;
        const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);

        if (!par->predictionStage)
        {
            const ForwardInput *in = static_cast<const ForwardInput * >(input);
            services::Collection<size_t> valueDim(1);
            valueDim[0] = 1;
            if (!get(layers::forward::value))
            {
                DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, valueDim);
            }

            if (!get(layers::forward::resultForBackward))
            {
                set(layers::forward::resultForBackward, layers::LayerDataPtr(new layers::LayerData()));
            }

            s |= setResultForBackward(input);
        }
        else
        {
            const ForwardInput *in = static_cast<const ForwardInput * >(input);
            TensorPtr dataTensor = in->get(layers::forward::data);
            DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), "data"));

            const services::Collection<size_t> &inputDims = dataTensor->getDimensions();
            DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, inputDims);
        }
        return s;
    }

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                    const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        return inputSize;
    }

    /**
     * Sets the result that is used in backward custom loss layer
     * \param[in] input     Pointer to an object containing the input data
     *
     * \return Status of computations
     */
    services::Status setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const ForwardInput *in = static_cast<const ForwardInput * >(input);
        set(auxData, in->get(layers::forward::data));
        set(auxGroundTruth, in->get(layers::loss::forward::groundTruth));
        return services::Status();
    }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<ForwardResult> ForwardResultPtr;

/**
 * \brief Provides methods to run implementations of the of the forward custom loss layer
 */
template<typename algorithmFPType>
class DAAL_EXPORT ForwardBatchContainer : public layers::forward::LayerContainerIfaceImpl
{
public:
    /**
    * Constructs a container for the forward custom loss layer with a specified environment
    * \param[in] daalEnv   Environment object
    */
    ForwardBatchContainer(daal::services::Environment::env *daalEnv) {}
    /** Default destructor */
    ~ForwardBatchContainer() {}
    /**
     * Computes the result of the forward custom loss layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * \brief Computes the result of the forward custom loss layer in the batch processing mode
 */
template<typename algorithmFPType = float>
class ForwardBatch : public layers::loss::forward::Batch
{
public:
    typedef layers::loss::forward::Batch super;

    typedef ForwardInput  InputType;
    typedef Parameter     ParameterType;
    typedef ForwardResult ResultType;

    ParameterType &parameter;  /*!< custom loss layer "parameters" structure */
    InputType input;           /*!< Input objects of the layer */

    /** Default constructor */
    ForwardBatch() : parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a forward custom loss layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    ForwardBatch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs a forward custom loss layer by copying input objects
     * and parameters of another forward custom loss layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    ForwardBatch(const ForwardBatch<algorithmFPType> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)0; }

    /**
     * Returns the structure that contains input objects of the custom forward layer
     * \return Structure that contains input objects of the custom forward layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the forward custom loss layer
     * \return Structure that contains parameters of the forward custom loss layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the forward custom loss layer
     * \return Structure that contains result of the forward custom loss layer
     */
    layers::forward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return getResult();
    }

    /**
     * Returns the structure that contains result of the forward custom loss layer
     * \return Structure that contains result of the forward custom loss layer
     */
    ForwardResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store result of the forward custom loss layer
     * \param[in] result  Structure to store result of the forward custom loss layer
     *
     * \return Status of computations
     */
    services::Status setResult(const ForwardResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated forward custom loss layer
     * with a copy of input objects and parameters of this forward custom loss layer
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<ForwardBatch<algorithmFPType> > clone() const
    {
        return services::SharedPtr<ForwardBatch<algorithmFPType> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the forward custom loss layer
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int)0);
        this->_res = this->_result.get();
        return s;
    }

    /**
     * Commented part is the code of the getLayerForPrediction() method from the forward logistic cross-entropy loss layer
     * It uses the forward logistic layer which is already implemented
     *
     * In this example we implement branching for training/prediction inside computation function,
     * so we don't need to create specific layer for prediction
     */
    /**
     * Returns forward custom layer for prediction - the layer that corresponds to this layer on the prediction stage
     * \return Forward custom layer for prediction
     */
/*
    virtual layers::forward::LayerIfacePtr getLayerForPrediction() const DAAL_C11_OVERRIDE
    {
        return layers::forward::LayerIfacePtr(
            new layers::logistic::forward::Batch<algorithmFPType>());
    }
*/

protected:
    virtual ForwardBatch<algorithmFPType> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new ForwardBatch<algorithmFPType>(*this);
    }

    void initialize()
    {
        daal::algorithms::Analysis<batch>::_ac = new ForwardBatchContainer<algorithmFPType>(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    ForwardResultPtr _result;
    ParameterType _defaultParameter;
};


/**
 * \brief Input objects for the backward custom loss layer
 */
class DAAL_EXPORT BackwardInput : public layers::loss::backward::Input
{
public:
    typedef layers::loss::backward::Input super;

    /** Default constructor */
    BackwardInput() {}

    /** Copy constructor */
    BackwardInput(const BackwardInput& other) : super(other) {}

    virtual ~BackwardInput() {}

    /**
    * Returns an input object for the backward custom loss layer
    * \param[in] id    Identifier of the input object
    * \return          Input object that corresponds to the given identifier
    */
    using layers::loss::backward::Input::get;

    /**
     * Sets an input object for the backward custom loss layer
     * \param[in] id      Identifier of the input object
     * \param[in] ptr     Pointer to the object
     */
    using layers::loss::backward::Input::set;

    /**
    * Returns an input object for the backward custom loss layer
    * \param[in] id    Identifier of the input object
    * \return          Input object that corresponds to the given identifier
    */
    data_management::TensorPtr get(LayerDataId id) const
    {
        layers::LayerDataPtr layerData =
            layers::LayerData::cast<SerializationIface>(Argument::get(layers::backward::inputFromForward));
        if(!layerData)
            return data_management::TensorPtr();
        return Tensor::cast<SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets an input object for the backward custom loss layer
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the object
     */
    void set(LayerDataId id, const data_management::TensorPtr &value)
    {
        layers::LayerDataPtr layerData =
            layers::LayerData::cast<SerializationIface>(Argument::get(layers::backward::inputFromForward));
        if(!layerData) return;
        (*layerData)[id] = value;
    }

    /**
    * Checks input object for the backward custom loss layer
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    *
     * \return Status of computation
    */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const layers::Parameter *parameter = static_cast<const Parameter *>(par);
        if (!parameter->propagateGradient) { return services::Status(); }

        services::Status s;
        DAAL_CHECK_STATUS(s, layers::loss::backward::Input::check(par, method));

        TensorPtr auxDataTensor = get(auxData);
        TensorPtr auxGroundTruthTensor = get(auxGroundTruth);

        DAAL_CHECK_STATUS(s, checkTensor(auxDataTensor.get(), "data"));
        const services::Collection<size_t> &inputDims = auxDataTensor->getDimensions();

        DAAL_CHECK_STATUS(s, checkTensor(auxGroundTruthTensor.get(), "groundTruth"));
        const services::Collection<size_t> &gtDims = auxGroundTruthTensor->getDimensions();

        DAAL_CHECK_EX(auxDataTensor->getSize() == auxGroundTruthTensor->getSize(), services::ErrorIncorrectSizeOfDimensionInTensor, services::ParameterName, "groundTruth");
        DAAL_CHECK_EX(gtDims.size() == 1 || gtDims.size() == inputDims.size() , services::ErrorIncorrectNumberOfDimensionsInTensor, services::ParameterName, "data");
        DAAL_CHECK_EX(gtDims[0] == inputDims[0] , services::ErrorIncorrectSizeOfDimensionInTensor, services::ParameterName, "data");
        return s;
    }
};

/**
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward custom loss layer
 */
class DAAL_EXPORT BackwardResult : public layers::loss::backward::Result
{
public:
    BackwardResult() : layers::loss::backward::Result() {}
    virtual ~BackwardResult() {};

    /**
     * Returns the result of the backward custom loss layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    using layers::loss::backward::Result::get;

    /**
     * Sets the result of the backward custom loss layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    using layers::loss::backward::Result::set;

    /**
     * Checks the result of the backward custom loss layer
     * \param[in] input   Input object for the algorithm
     * \param[in] par     Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computation
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const layers::Parameter *parameter = static_cast<const Parameter *>(par);
        if (!parameter->propagateGradient) { return services::Status(); }

        const BackwardInput *algInput = static_cast<const BackwardInput *>(input);

        const services::Collection<size_t> &gradDims = algInput->get(auxData)->getDimensions();
        return checkTensor(get(layers::backward::gradient).get(), "gradient", &gradDims);
    }

    /**
     * Allocates memory to store the result of the backward custom loss layer
     * \param[in] input     Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method
     *
     * \return Status of computation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const layers::Parameter *param = static_cast<const Parameter *>(parameter);
        if (!param->propagateGradient) { return services::Status(); }

        services::Status s;
        if (!get(layers::backward::gradient))
        {
            const BackwardInput *in = static_cast<const BackwardInput *>(input);

            data_management::TensorPtr probabilitiesTable = in->get(auxData);

            DAAL_CHECK_EX(probabilitiesTable.get(), services::ErrorNullInputNumericTable, services::ArgumentName, "auxData");

            DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, probabilitiesTable->getDimensions());
        }
        return s;
    }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<BackwardResult> BackwardResultPtr;

/**
 * \brief Provides methods to run implementations of the of the backward custom loss layer
 */
template<typename algorithmFPType>
class DAAL_EXPORT BackwardBatchContainer : public algorithms::AnalysisContainerIface<batch>
{
public:
    /**
    * Constructs a container for the backward custom loss layer with a specified environment
    * \param[in] daalEnv   Environment object
    */
    BackwardBatchContainer(daal::services::Environment::env *daalEnv) {}
    /** Default destructor */
    ~BackwardBatchContainer() {}
    /**
     * Computes the result of the backward custom loss layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * \brief Computes the results of the backward custom loss layer in the batch processing mode
 */
template<typename algorithmFPType = float>
class BackwardBatch : public layers::loss::backward::Batch
{
public:
    typedef layers::loss::backward::Batch super;

    typedef BackwardInput  InputType;
    typedef Parameter      ParameterType;
    typedef BackwardResult ResultType;

    ParameterType &parameter; /*!< custom loss layer "parameters" structure */
    InputType input;          /*!< Input objects of the layer */

    /** Default constructor */
    BackwardBatch() : parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a backward custom loss layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    BackwardBatch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs backward custom loss layer by copying input objects
     * and parameters of another backward custom loss layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    BackwardBatch(const BackwardBatch<algorithmFPType> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)0; }

    /**
     * Returns the structure that contains input objects of the custom backward layer
     * \return Structure that contains input objects of the custom backward layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains prameters of the backward custom loss layer
     * \return Structure that contains parameters of the backward custom loss layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the backward custom loss layer
     * \return Structure that contains result of the backward custom loss layer
     */
    layers::backward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return _result;
    }

    /**
     * Returns the structure that contains result of the backward custom loss layer
     * \return Structure that contains result of the backward custom loss layer
     */
    BackwardResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the backward custom loss layer
     * \param[in] result  Structure to store result of the backward custom loss layer
     *
     * \return Status of computations
     */
    services::Status setResult(const BackwardResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated the backward custom loss layer
     * with a copy of input objects and parameters of this backward custom loss layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<BackwardBatch<algorithmFPType> > clone() const
    {
        return services::SharedPtr<BackwardBatch<algorithmFPType> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the backward custom loss layer
    *
     * \return Status of computations
    */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int)0);
        this->_res = this->_result.get();
        return s;
    }

protected:
    virtual BackwardBatch<algorithmFPType> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new BackwardBatch<algorithmFPType>(*this);
    }

    void initialize()
    {
        daal::algorithms::Analysis<batch>::_ac = new BackwardBatchContainer<algorithmFPType>(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    BackwardResultPtr _result;
    ParameterType _defaultParameter;
};

/**
 * \brief Provides methods for the custom loss layer in the batch processing mode
 */
template<typename algorithmFPType = float>
class Batch : public layers::loss::Batch
{
public:
    Parameter parameter; /*!< custom loss layer parameters */
    /** Default constructor */
    Batch()
    {
        ForwardBatch<algorithmFPType> *forwardLayerObject = new ForwardBatch<algorithmFPType>(parameter);
        BackwardBatch<algorithmFPType> *backwardLayerObject = new BackwardBatch<algorithmFPType>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<ForwardBatch<algorithmFPType> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<BackwardBatch<algorithmFPType> >(backwardLayerObject);
    };
};

/**
 * Implementation of the algorithm for computation of the custom forward loss layer
 */
template<typename algorithmFPType>
daal::services::Status ForwardBatchContainer<algorithmFPType>::compute()
{
    daal::services::Environment::env &env = *_env;

    ForwardInput *input = static_cast<ForwardInput *>(_in);
    ForwardResult *result = static_cast<ForwardResult *>(_res);

    Parameter *parameter = static_cast<Parameter *>(_par);
    if (!parameter->predictionStage)
    {
        daal::data_management::Tensor *inputTensor         = input->get(layers::forward::data).get();
        daal::data_management::Tensor *groundTruthTensor   = input->get(layers::loss::forward::groundTruth).get();
        daal::data_management::Tensor *resultTensor        = result->get(layers::forward::value).get();

        size_t nRowsToProcess = inputTensor->getDimensionSize(0);

        daal::data_management::SubtensorDescriptor<algorithmFPType> inputBlock;
        inputTensor->getSubtensor(0, 0, 0, nRowsToProcess, readOnly, inputBlock);
        const algorithmFPType *inputArray = inputBlock.getPtr();

        daal::data_management::SubtensorDescriptor<algorithmFPType> groundTruthBlock;
        groundTruthTensor->getSubtensor(0, 0, 0, nRowsToProcess, readOnly, groundTruthBlock);
        const algorithmFPType *groundTruthArray = groundTruthBlock.getPtr();

        daal::data_management::SubtensorDescriptor<algorithmFPType> resultBlock;
        resultTensor->getSubtensor(0, 0, 0, nRowsToProcess, readWrite, resultBlock);
        algorithmFPType &loss = resultBlock.getPtr()[0];

        size_t nDataElements = inputBlock.getSize();
        loss = 0;
        for(size_t i = 0; i < nDataElements; i++)
        {
            loss += (inputArray[i] - groundTruthArray[i]) * (inputArray[i] - groundTruthArray[i]);
        }
        loss = loss / nRowsToProcess;

        resultTensor->releaseSubtensor(resultBlock);
    }
    else
    {
        daal::data_management::Tensor *inputTensor  = input->get(layers::forward::data).get();
        daal::data_management::Tensor *resultTensor = result->get(layers::forward::value).get();

        size_t nRowsToProcess = inputTensor->getDimensionSize(0);

        daal::data_management::SubtensorDescriptor<algorithmFPType> inputBlock;
        inputTensor->getSubtensor(0, 0, 0, nRowsToProcess, readOnly, inputBlock);
        const algorithmFPType *inputArray = inputBlock.getPtr();

        daal::data_management::SubtensorDescriptor<algorithmFPType> resultBlock;
        resultTensor->getSubtensor(0, 0, 0, nRowsToProcess, readWrite, resultBlock);
        algorithmFPType *resultArray = resultBlock.getPtr();

        size_t nDataElements = inputBlock.getSize();
        for(size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = inputArray[i];
        }

        resultTensor->releaseSubtensor(resultBlock);
    }
    return services::Status();
}

/**
 * Implementation of the algorithm for computation of the custom backward loss layer
 */
template<typename algorithmFPType>
daal::services::Status BackwardBatchContainer<algorithmFPType>::compute()
{
    daal::services::Environment::env &env = *_env;

    BackwardInput *input = static_cast<BackwardInput *>(_in);
    BackwardResult *result = static_cast<BackwardResult *>(_res);

    Parameter *parameter = static_cast<Parameter *>(_par);
    if (!parameter->propagateGradient) { return services::Status(); }

    daal::data_management::Tensor *inputTensor       = input->get(auxData).get();
    daal::data_management::Tensor *groundTruthTensor = input->get(auxGroundTruth).get();
    daal::data_management::Tensor *resultTensor      = result->get(layers::backward::gradient).get();

    size_t nRowsToProcess = inputTensor->getDimensionSize(0);

    daal::data_management::SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, 0, nRowsToProcess, readOnly, inputBlock);
    const algorithmFPType *inputArray = inputBlock.getPtr();

    daal::data_management::SubtensorDescriptor<algorithmFPType> groundTruthBlock;
    groundTruthTensor->getSubtensor(0, 0, 0, nRowsToProcess, readOnly, groundTruthBlock);
    const algorithmFPType *groundTruthArray = groundTruthBlock.getPtr();

    daal::data_management::SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, 0, nRowsToProcess, readWrite, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();
    algorithmFPType invBatchSize = (algorithmFPType)1.0 / (inputTensor->getDimensionSize(0));
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = invBatchSize * (2 * (inputArray[i] - groundTruthArray[i]));
    }

    resultTensor->releaseSubtensor(resultBlock);

    return services::Status();
}

}
