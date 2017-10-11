%module daal

%include "typemaps_data.i"
%include "except.i"

%{
#include "hlapi_distr.h"

static std::map< std::string, int > string2enum_algorithms__multinomial_naive_bayes__prediction =
{
    {"defaultDense", algorithms::multinomial_naive_bayes::prediction::defaultDense},
    {"fastCSR", algorithms::multinomial_naive_bayes::prediction::fastCSR},
};

static std::map< std::string, int > string2enum_algorithms__multinomial_naive_bayes__training =
{
    {"defaultDense", algorithms::multinomial_naive_bayes::training::defaultDense},
    {"fastCSR", algorithms::multinomial_naive_bayes::training::fastCSR},
    {"partialModels", algorithms::multinomial_naive_bayes::training::partialModels},
};

static std::map< std::string, int > string2enum_algorithms__kmeans__init =
{
    {"deterministicDense", algorithms::kmeans::init::deterministicDense},
    {"defaultDense", algorithms::kmeans::init::defaultDense},
    {"randomDense", algorithms::kmeans::init::randomDense},
    {"plusPlusDense", algorithms::kmeans::init::plusPlusDense},
    {"parallelPlusDense", algorithms::kmeans::init::parallelPlusDense},
    {"deterministicCSR", algorithms::kmeans::init::deterministicCSR},
    {"randomCSR", algorithms::kmeans::init::randomCSR},
    {"plusPlusCSR", algorithms::kmeans::init::plusPlusCSR},
    {"parallelPlusCSR", algorithms::kmeans::init::parallelPlusCSR},
    {"data", algorithms::kmeans::init::data},
    {"partialResults", algorithms::kmeans::init::partialResults},
    {"internalInput", algorithms::kmeans::init::internalInput},
    {"inputOfStep2", algorithms::kmeans::init::inputOfStep2},
    {"inputOfStep3FromStep2", algorithms::kmeans::init::inputOfStep3FromStep2},
    {"inputOfStep4FromStep3", algorithms::kmeans::init::inputOfStep4FromStep3},
    {"inputCentroids", algorithms::kmeans::init::inputCentroids},
    {"inputOfStep5FromStep2", algorithms::kmeans::init::inputOfStep5FromStep2},
    {"inputOfStep5FromStep3", algorithms::kmeans::init::inputOfStep5FromStep3},
    {"partialCentroids", algorithms::kmeans::init::partialCentroids},
    {"partialClusters", algorithms::kmeans::init::partialClusters},
    {"partialClustersNumber", algorithms::kmeans::init::partialClustersNumber},
    {"outputOfStep2ForStep3", algorithms::kmeans::init::outputOfStep2ForStep3},
    {"outputOfStep2ForStep5", algorithms::kmeans::init::outputOfStep2ForStep5},
    {"internalResult", algorithms::kmeans::init::internalResult},
    {"outputOfStep3ForStep4", algorithms::kmeans::init::outputOfStep3ForStep4},
    {"rngState", algorithms::kmeans::init::rngState},
    {"outputOfStep3ForStep5", algorithms::kmeans::init::outputOfStep3ForStep5},
    {"outputOfStep4", algorithms::kmeans::init::outputOfStep4},
    {"candidates", algorithms::kmeans::init::candidates},
    {"weights", algorithms::kmeans::init::weights},
    {"centroids", algorithms::kmeans::init::centroids},
};

static std::map< std::string, int > string2enum_algorithms__kmeans =
{
    {"lloydDense", algorithms::kmeans::lloydDense},
    {"defaultDense", algorithms::kmeans::defaultDense},
    {"lloydCSR", algorithms::kmeans::lloydCSR},
    {"euclidean", algorithms::kmeans::euclidean},
    {"data", algorithms::kmeans::data},
    {"inputCentroids", algorithms::kmeans::inputCentroids},
    {"partialResults", algorithms::kmeans::partialResults},
    {"nObservations", algorithms::kmeans::nObservations},
    {"partialSums", algorithms::kmeans::partialSums},
    {"partialObjectiveFunction", algorithms::kmeans::partialObjectiveFunction},
    {"partialGoalFunction", algorithms::kmeans::partialGoalFunction},
    {"partialAssignments", algorithms::kmeans::partialAssignments},
    {"centroids", algorithms::kmeans::centroids},
    {"assignments", algorithms::kmeans::assignments},
    {"objectiveFunction", algorithms::kmeans::objectiveFunction},
    {"goalFunction", algorithms::kmeans::goalFunction},
    {"nIterations", algorithms::kmeans::nIterations},
};

static std::map< std::string, int > string2enum_algorithms__pca =
{
    {"correlationDense", algorithms::pca::correlationDense},
    {"defaultDense", algorithms::pca::defaultDense},
    {"svdDense", algorithms::pca::svdDense},
    {"data", algorithms::pca::data},
    {"correlation", algorithms::pca::correlation},
    {"partialResults", algorithms::pca::partialResults},
    {"nObservationsCorrelation", algorithms::pca::nObservationsCorrelation},
    {"crossProductCorrelation", algorithms::pca::crossProductCorrelation},
    {"sumCorrelation", algorithms::pca::sumCorrelation},
    {"nObservationsSVD", algorithms::pca::nObservationsSVD},
    {"sumSVD", algorithms::pca::sumSVD},
    {"sumSquaresSVD", algorithms::pca::sumSquaresSVD},
    {"auxiliaryData", algorithms::pca::auxiliaryData},
    {"distributedInputs", algorithms::pca::distributedInputs},
    {"eigenvalues", algorithms::pca::eigenvalues},
    {"eigenvectors", algorithms::pca::eigenvectors},
};

static std::map< std::string, int > string2enum_algorithms__kernel_function__linear =
{
    {"defaultDense", algorithms::kernel_function::linear::defaultDense},
    {"fastCSR", algorithms::kernel_function::linear::fastCSR},
};

static std::map< std::string, int > string2enum_algorithms__kernel_function =
{
    {"vectorVector", algorithms::kernel_function::vectorVector},
    {"matrixVector", algorithms::kernel_function::matrixVector},
    {"matrixMatrix", algorithms::kernel_function::matrixMatrix},
    {"X", algorithms::kernel_function::X},
    {"Y", algorithms::kernel_function::Y},
    {"values", algorithms::kernel_function::values},
};

static std::map< std::string, int > string2enum_algorithms__kernel_function__rbf =
{
    {"defaultDense", algorithms::kernel_function::rbf::defaultDense},
    {"fastCSR", algorithms::kernel_function::rbf::fastCSR},
};

static std::map< std::string, int > string2enum_algorithms__svm__training =
{
    {"boser", algorithms::svm::training::boser},
    {"defaultDense", algorithms::svm::training::defaultDense},
};

static std::map< std::string, int > string2enum_algorithms__svm__prediction =
{
    {"defaultDense", algorithms::svm::prediction::defaultDense},
};

static std::map< std::string, int > string2enum_algorithms__linear_regression__prediction =
{
    {"defaultDense", algorithms::linear_regression::prediction::defaultDense},
    {"data", algorithms::linear_regression::prediction::data},
    {"model", algorithms::linear_regression::prediction::model},
    {"prediction", algorithms::linear_regression::prediction::prediction},
};

static std::map< std::string, int > string2enum_algorithms__linear_regression__training =
{
    {"defaultDense", algorithms::linear_regression::training::defaultDense},
    {"normEqDense", algorithms::linear_regression::training::normEqDense},
    {"qrDense", algorithms::linear_regression::training::qrDense},
    {"data", algorithms::linear_regression::training::data},
    {"dependentVariables", algorithms::linear_regression::training::dependentVariables},
    {"partialModels", algorithms::linear_regression::training::partialModels},
    {"partialModel", algorithms::linear_regression::training::partialModel},
    {"model", algorithms::linear_regression::training::model},
};

static std::map< std::string, int > string2enum_algorithms__univariate_outlier_detection =
{
    {"defaultDense", algorithms::univariate_outlier_detection::defaultDense},
    {"data", algorithms::univariate_outlier_detection::data},
    {"location", algorithms::univariate_outlier_detection::location},
    {"scatter", algorithms::univariate_outlier_detection::scatter},
    {"threshold", algorithms::univariate_outlier_detection::threshold},
    {"weights", algorithms::univariate_outlier_detection::weights},
};

static std::map< std::string, int > string2enum_algorithms__multivariate_outlier_detection =
{
    {"defaultDense", algorithms::multivariate_outlier_detection::defaultDense},
    {"baconDense", algorithms::multivariate_outlier_detection::baconDense},
    {"baconMedian", algorithms::multivariate_outlier_detection::baconMedian},
    {"baconMahalanobis", algorithms::multivariate_outlier_detection::baconMahalanobis},
    {"data", algorithms::multivariate_outlier_detection::data},
    {"location", algorithms::multivariate_outlier_detection::location},
    {"scatter", algorithms::multivariate_outlier_detection::scatter},
    {"threshold", algorithms::multivariate_outlier_detection::threshold},
    {"weights", algorithms::multivariate_outlier_detection::weights},
};

static std::map< std::string, int > string2enum_algorithms__svd =
{
    {"defaultDense", algorithms::svd::defaultDense},
    {"notRequired", algorithms::svd::notRequired},
    {"requiredInPackedForm", algorithms::svd::requiredInPackedForm},
    {"data", algorithms::svd::data},
    {"singularValues", algorithms::svd::singularValues},
    {"leftSingularMatrix", algorithms::svd::leftSingularMatrix},
    {"rightSingularMatrix", algorithms::svd::rightSingularMatrix},
    {"outputOfStep1ForStep3", algorithms::svd::outputOfStep1ForStep3},
    {"outputOfStep1ForStep2", algorithms::svd::outputOfStep1ForStep2},
    {"outputOfStep2ForStep3", algorithms::svd::outputOfStep2ForStep3},
    {"finalResultFromStep2Master", algorithms::svd::finalResultFromStep2Master},
    {"finalResultFromStep3", algorithms::svd::finalResultFromStep3},
    {"inputOfStep2FromStep1", algorithms::svd::inputOfStep2FromStep1},
    {"inputOfStep3FromStep1", algorithms::svd::inputOfStep3FromStep1},
    {"inputOfStep3FromStep2", algorithms::svd::inputOfStep3FromStep2},
};

static std::map< std::string, int > string2enum_algorithms__multi_class_classifier__training =
{
    {"oneAgainstOne", algorithms::multi_class_classifier::training::oneAgainstOne},
};

static std::map< std::string, int > string2enum_algorithms__multi_class_classifier__prediction =
{
    {"defaultDense", algorithms::multi_class_classifier::prediction::defaultDense},
    {"multiClassClassifierWu", algorithms::multi_class_classifier::prediction::multiClassClassifierWu},
};

%}

// Warning: no result found for algorithms::multinomial_naive_bayes
//Warning: parameter member algorithms::kmeans::init::engine of algorithms::kmeans::init is no stdtype, no enum and has no typemap. Ignored.
//Warning: do not know what to do with algorithms::kmeans algorithms::kmeans::distanceType
//Warning: no members of "parameter" found for ['algorithms::pca', 'BatchParameter<algorithmFPType, method>']: {'InputIface': <parse.cpp_class object at 0x2b9f9f8d94e0>, 'Input': <parse.cpp_class object at 0x2b9f9f8d9588>, 'PartialResultBase': <parse.cpp_class object at 0x2b9f9f8d9710>, 'PartialResult': <parse.cpp_class object at 0x2b9f9f8d9780>, 'PartialResult<daal::algorithms::pca::correlationDense>': <parse.cpp_class object at 0x2b9f9f8d9860>, 'PartialResult<daal::algorithms::pca::svdDense>': <parse.cpp_class object at 0x2b9f9f8d9a20>, 'BaseParameter': <parse.cpp_class object at 0x2b9f9f8d9c18>, 'BatchParameter': <parse.cpp_class object at 0x2b9f9f8d9c88>, 'BatchParameter<correlationDense>': <parse.cpp_class object at 0x2b9f9f8d9d68>, 'OnlineParameter': <parse.cpp_class object at 0x2b9f9f8d9da0>, 'OnlineParameter<correlationDense>': <parse.cpp_class object at 0x2b9f9f8d9e80>, 'OnlineParameter<svdDense>': <parse.cpp_class object at 0x2b9f9f8d9eb8>, 'DistributedParameter': <parse.cpp_class object at 0x2b9f9f8d9ef0>, 'DistributedParameter<step2Master, correlationDense>': <parse.cpp_class object at 0x2b9f9f8d9fd0>, 'DistributedInput': <parse.cpp_class object at 0x2b9f9f8e7048>, 'DistributedInput<correlationDense>': <parse.cpp_class object at 0x2b9f9f8e7128>, 'DistributedInput<svdDense>': <parse.cpp_class object at 0x2b9f9f8e7278>, 'Result': <parse.cpp_class object at 0x2b9f9f8e73c8>, 'Online': <parse.cpp_class object at 0x2b9f9f8e75c0>, 'Online<correlationDense>': <parse.cpp_class object at 0x2b9f9f8e75f8>, 'Online<svdDense>': <parse.cpp_class object at 0x2b9f9f8e76d8>, 'Batch': <parse.cpp_class object at 0x2b9f9f8e7860>, 'Distributed': <parse.cpp_class object at 0x2b9f9f8e7940>, 'Distributed<step1Local>': <parse.cpp_class object at 0x2b9f9f8e79b0>, 'Distributed<step2Master, correlationDense>': <parse.cpp_class object at 0x2b9f9f8e7a90>, 'Distributed<step2Master, svdDense>': <parse.cpp_class object at 0x2b9f9f8e7b38>}

// Warning: no result found for algorithms::kernel_function
// Warning: result typemap already defined for algorithms::kernel_function::rbf
// Warning: no result found for algorithms::svm
// Warning: result typemap already defined for algorithms::svm::prediction
// Warning: no result found for algorithms::linear_regression
//Warning: no parameter_type found for algorithms::linear_regression::prediction::Batch

//Warning: no parameter_type found for algorithms::univariate_outlier_detection::Batch

//Warning: no parameter_type found for algorithms::multivariate_outlier_detection::Batch

// Warning: no result found for algorithms::multi_class_classifier
// Warning: result typemap already defined for algorithms::multi_class_classifier::prediction
// Warning: no result found for algorithms::classifier

%rename("%(regex:/.*::run_step.*/\"$ignore\"/)s") "";
%warnfilter(401) algo_manager_i;


%include "daal_shared_ptr.i";


%shared_ptr(KernelIface_i);
%ignore *::get_KernelIfacePtr;
%inline %{
class KernelIface_i : public algo_manager_i
{
public:
    typedef daal::algorithms::kernel_function::KernelIfacePtr KernelIfacePtr_type;
    virtual KernelIfacePtr_type get_KernelIfacePtr()
    {
        return KernelIfacePtr_type();
    }
};
%}
%typemap(in)
(const daal::algorithms::kernel_function::KernelIfacePtr)
{
    void *argp = 0;
    int newmem = 0;
    int res = SWIG_ConvertPtrAndOwn($input, &argp, SWIGTYPE_p_daal__services__SharedPtrT_KernelIface_i_t, %convertptr_flags, &newmem);
    if (!SWIG_IsOK(res)) {
        %argument_fail(res, "$type", $symname, $argnum);
    }
    if(argp) {
        daal::services::SharedPtr< KernelIface_i > tmp_kim(*(%reinterpret_cast(argp, daal::services::SharedPtr< KernelIface_i >*)));
        if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, daal::services::SharedPtr< KernelIface_i >*);
        $1 = tmp_kim->get_KernelIfacePtr();
    } else {
        $1.reset();
    }
}


%shared_ptr(classifier_prediction_Batch_i);
%ignore *::get_classifier_prediction_BatchPtr;
%inline %{
class classifier_prediction_Batch_i : public algo_manager_i
{
public:
    typedef daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> classifier_prediction_BatchPtr_type;
    virtual classifier_prediction_BatchPtr_type get_classifier_prediction_BatchPtr()
    {
        return classifier_prediction_BatchPtr_type();
    }
};
%}
%typemap(in)
(const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>)
{
    void *argp = 0;
    int newmem = 0;
    int res = SWIG_ConvertPtrAndOwn($input, &argp, SWIGTYPE_p_daal__services__SharedPtrT_classifier_prediction_Batch_i_t, %convertptr_flags, &newmem);
    if (!SWIG_IsOK(res)) {
        %argument_fail(res, "$type", $symname, $argnum);
    }
    if(argp) {
        daal::services::SharedPtr< classifier_prediction_Batch_i > tmp_kim(*(%reinterpret_cast(argp, daal::services::SharedPtr< classifier_prediction_Batch_i >*)));
        if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, daal::services::SharedPtr< classifier_prediction_Batch_i >*);
        $1 = tmp_kim->get_classifier_prediction_BatchPtr();
    } else {
        $1.reset();
    }
}


%shared_ptr(classifier_training_Batch_i);
%ignore *::get_classifier_training_BatchPtr;
%inline %{
class classifier_training_Batch_i : public algo_manager_i
{
public:
    typedef daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> classifier_training_BatchPtr_type;
    virtual classifier_training_BatchPtr_type get_classifier_training_BatchPtr()
    {
        return classifier_training_BatchPtr_type();
    }
};
%}
%typemap(in)
(const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>)
{
    void *argp = 0;
    int newmem = 0;
    int res = SWIG_ConvertPtrAndOwn($input, &argp, SWIGTYPE_p_daal__services__SharedPtrT_classifier_training_Batch_i_t, %convertptr_flags, &newmem);
    if (!SWIG_IsOK(res)) {
        %argument_fail(res, "$type", $symname, $argnum);
    }
    if(argp) {
        daal::services::SharedPtr< classifier_training_Batch_i > tmp_kim(*(%reinterpret_cast(argp, daal::services::SharedPtr< classifier_training_Batch_i >*)));
        if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, daal::services::SharedPtr< classifier_training_Batch_i >*);
        $1 = tmp_kim->get_classifier_training_BatchPtr();
    } else {
        $1.reset();
    }
}

%shared_ptr(multinomial_naive_bayes_prediction_i);
%shared_ptr(multinomial_naive_bayes_training_i);
%shared_ptr(kmeans_init_i);
%shared_ptr(kmeans_i);
%shared_ptr(pca_i);
%shared_ptr(linear_i);
%shared_ptr(rbf_i);
%shared_ptr(svm_training_i);
%shared_ptr(svm_prediction_i);
%shared_ptr(linear_regression_prediction_i);
%shared_ptr(linear_regression_training_i);
%shared_ptr(univariate_outlier_detection_i);
%shared_ptr(multivariate_outlier_detection_i);
%shared_ptr(svd_i);
%shared_ptr(multi_class_classifier_training_i);
%shared_ptr(multi_class_classifier_prediction_i);

%inline %{

class multinomial_naive_bayes_prediction_i : public classifier_prediction_Batch_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::algorithms::multinomial_naive_bayes::ModelPtr i_model) = 0;
};

class multinomial_naive_bayes_training_i : public classifier_training_Batch_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const TableOrFList & i_labels) = 0;
};

class kmeans_init_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data) = 0;
};

class kmeans_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::data_management::NumericTablePtr i_inputCentroids) = 0;
};

class pca_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data) = 0;
};

class linear_i : public KernelIface_i
{
public:
    virtual NTYPE compute(const daal::data_management::NumericTablePtr i_X,
                          const daal::data_management::NumericTablePtr i_Y) = 0;
};

class rbf_i : public KernelIface_i
{
public:
    virtual NTYPE compute(const daal::data_management::NumericTablePtr i_X,
                          const daal::data_management::NumericTablePtr i_Y) = 0;
};

class svm_training_i : public classifier_training_Batch_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::data_management::NumericTablePtr i_labels) = 0;
};

class svm_prediction_i : public classifier_prediction_Batch_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::algorithms::svm::ModelPtr i_model) = 0;
};

class linear_regression_prediction_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::algorithms::linear_regression::ModelPtr i_model) = 0;
};

class linear_regression_training_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::data_management::NumericTablePtr i_dependentVariables) = 0;
};

class univariate_outlier_detection_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::data_management::NumericTablePtr i_location = data_management::NumericTablePtr(),
                          const daal::data_management::NumericTablePtr i_scatter = data_management::NumericTablePtr(),
                          const daal::data_management::NumericTablePtr i_threshold = data_management::NumericTablePtr()) = 0;
};

class multivariate_outlier_detection_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::data_management::NumericTablePtr i_location = data_management::NumericTablePtr(),
                          const daal::data_management::NumericTablePtr i_scatter = data_management::NumericTablePtr(),
                          const daal::data_management::NumericTablePtr i_threshold = data_management::NumericTablePtr()) = 0;
};

class svd_i : public algo_manager_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data) = 0;
};

class multi_class_classifier_training_i : public classifier_training_Batch_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::data_management::NumericTablePtr i_labels) = 0;
};

class multi_class_classifier_prediction_i : public classifier_prediction_Batch_i
{
public:
    virtual NTYPE compute(const TableOrFList & i_data,
                          const daal::algorithms::multi_class_classifier::ModelPtr i_model) = 0;
};

%}



%{

NTYPE native_type(daal::algorithms::classifier::ModelPtr & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "NFeatures", "NumberOfFeatures", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    size_t tmpNFeatures = obj_->getNFeatures();
    SET_ELT(res_, 0, native_type(tmpNFeatures, gc), names);
    size_t tmpNumberOfFeatures = obj_->getNumberOfFeatures();
    SET_ELT(res_, 1, native_type(tmpNumberOfFeatures, gc), names);
    MK_DAALPTR(dp_, new daal::algorithms::classifier::ModelPtr(obj_), daal::algorithms::classifier::ModelPtr, gc);
    SET_ELT(res_, 2, dp_, names);

    return res_;
}

NTYPE native_type(daal::algorithms::multi_class_classifier::ModelPtr & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "MultiClassClassifierModel", "NumberOfTwoClassClassifierModels", "NumberOfFeatures", "NFeatures", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    data_management::DataCollectionPtr tmpMultiClassClassifierModel = obj_->getMultiClassClassifierModel();
    SET_ELT(res_, 0, native_type(tmpMultiClassClassifierModel, gc), names);
    size_t tmpNumberOfTwoClassClassifierModels = obj_->getNumberOfTwoClassClassifierModels();
    SET_ELT(res_, 1, native_type(tmpNumberOfTwoClassClassifierModels, gc), names);
    size_t tmpNumberOfFeatures = obj_->getNumberOfFeatures();
    SET_ELT(res_, 2, native_type(tmpNumberOfFeatures, gc), names);
    size_t tmpNFeatures = obj_->getNFeatures();
    SET_ELT(res_, 3, native_type(tmpNFeatures, gc), names);
    MK_DAALPTR(dp_, new daal::algorithms::multi_class_classifier::ModelPtr(obj_), daal::algorithms::multi_class_classifier::ModelPtr, gc);
    SET_ELT(res_, 4, dp_, names);

    return res_;
}

NTYPE native_type(daal::algorithms::linear_regression::ModelPtr & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "NumberOfBetas", "NumberOfResponses", "InterceptFlag", "Beta", "NumberOfFeatures", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    size_t tmpNumberOfBetas = obj_->getNumberOfBetas();
    SET_ELT(res_, 0, native_type(tmpNumberOfBetas, gc), names);
    size_t tmpNumberOfResponses = obj_->getNumberOfResponses();
    SET_ELT(res_, 1, native_type(tmpNumberOfResponses, gc), names);
    bool tmpInterceptFlag = obj_->getInterceptFlag();
    SET_ELT(res_, 2, native_type(tmpInterceptFlag, gc), names);
    data_management::NumericTablePtr tmpBeta = obj_->getBeta();
    SET_ELT(res_, 3, native_type(tmpBeta, gc), names);
    size_t tmpNumberOfFeatures = obj_->getNumberOfFeatures();
    SET_ELT(res_, 4, native_type(tmpNumberOfFeatures, gc), names);
    MK_DAALPTR(dp_, new daal::algorithms::linear_regression::ModelPtr(obj_), daal::algorithms::linear_regression::ModelPtr, gc);
    SET_ELT(res_, 5, dp_, names);

    return res_;
}

NTYPE native_type(daal::algorithms::svm::ModelPtr & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "SupportVectors", "ClassificationCoefficients", "Bias", "NumberOfFeatures", "NFeatures", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    data_management::NumericTablePtr tmpSupportVectors = obj_->getSupportVectors();
    SET_ELT(res_, 0, native_type(tmpSupportVectors, gc), names);
    data_management::NumericTablePtr tmpClassificationCoefficients = obj_->getClassificationCoefficients();
    SET_ELT(res_, 1, native_type(tmpClassificationCoefficients, gc), names);
    double tmpBias = obj_->getBias();
    SET_ELT(res_, 2, native_type(tmpBias, gc), names);
    size_t tmpNumberOfFeatures = obj_->getNumberOfFeatures();
    SET_ELT(res_, 3, native_type(tmpNumberOfFeatures, gc), names);
    size_t tmpNFeatures = obj_->getNFeatures();
    SET_ELT(res_, 4, native_type(tmpNFeatures, gc), names);
    MK_DAALPTR(dp_, new daal::algorithms::svm::ModelPtr(obj_), daal::algorithms::svm::ModelPtr, gc);
    SET_ELT(res_, 5, dp_, names);

    return res_;
}

NTYPE native_type(daal::algorithms::multinomial_naive_bayes::ModelPtr & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "LogP", "LogTheta", "AuxTable", "NumberOfFeatures", "NFeatures", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    data_management::NumericTablePtr tmpLogP = obj_->getLogP();
    SET_ELT(res_, 0, native_type(tmpLogP, gc), names);
    data_management::NumericTablePtr tmpLogTheta = obj_->getLogTheta();
    SET_ELT(res_, 1, native_type(tmpLogTheta, gc), names);
    data_management::NumericTablePtr tmpAuxTable = obj_->getAuxTable();
    SET_ELT(res_, 2, native_type(tmpAuxTable, gc), names);
    size_t tmpNumberOfFeatures = obj_->getNumberOfFeatures();
    SET_ELT(res_, 3, native_type(tmpNumberOfFeatures, gc), names);
    size_t tmpNFeatures = obj_->getNFeatures();
    SET_ELT(res_, 4, native_type(tmpNFeatures, gc), names);
    MK_DAALPTR(dp_, new daal::algorithms::multinomial_naive_bayes::ModelPtr(obj_), daal::algorithms::multinomial_naive_bayes::ModelPtr, gc);
    SET_ELT(res_, 5, dp_, names);

    return res_;
}

// *****************************************
// algorithms::multinomial_naive_bayes

// *****************************************
// algorithms::multinomial_naive_bayes::prediction

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::classifier::prediction::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "prediction", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpprediction = (obj_)->get(algorithms::classifier::prediction::prediction);
    SET_ELT(res_, 0, native_type(tmpprediction, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::classifier::prediction::Result >(obj_), daal::services::SharedPtr< daal::algorithms::classifier::prediction::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::multinomial_naive_bayes::training

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::multinomial_naive_bayes::training::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "model", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::algorithms::multinomial_naive_bayes::ModelPtr tmpmodel = (obj_)->get(algorithms::classifier::training::model);
    SET_ELT(res_, 0, native_type(tmpmodel, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::multinomial_naive_bayes::training::Result >(obj_), daal::services::SharedPtr< daal::algorithms::multinomial_naive_bayes::training::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::kmeans::init

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::kmeans::init::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "centroids", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpcentroids = (obj_)->get(algorithms::kmeans::init::centroids);
    SET_ELT(res_, 0, native_type(tmpcentroids, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::kmeans::init::Result >(obj_), daal::services::SharedPtr< daal::algorithms::kmeans::init::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::kmeans

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::kmeans::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "centroids", "assignments", "objectiveFunction", "goalFunction", "nIterations", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpcentroids = (obj_)->get(algorithms::kmeans::centroids);
    SET_ELT(res_, 0, native_type(tmpcentroids, gc), names);
    daal::data_management::NumericTablePtr tmpassignments = (obj_)->get(algorithms::kmeans::assignments);
    SET_ELT(res_, 1, native_type(tmpassignments, gc), names);
    daal::data_management::NumericTablePtr tmpobjectiveFunction = (obj_)->get(algorithms::kmeans::objectiveFunction);
    SET_ELT(res_, 2, native_type(tmpobjectiveFunction, gc), names);
    daal::data_management::NumericTablePtr tmpgoalFunction = (obj_)->get(algorithms::kmeans::goalFunction);
    SET_ELT(res_, 3, native_type(tmpgoalFunction, gc), names);
    daal::data_management::NumericTablePtr tmpnIterations = (obj_)->get(algorithms::kmeans::nIterations);
    SET_ELT(res_, 4, native_type(tmpnIterations, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::kmeans::Result >(obj_), daal::services::SharedPtr< daal::algorithms::kmeans::Result >, gc);
    SET_ELT(res_, 5, dp_, names);

    return res_;
}

// *****************************************
// algorithms::pca

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::pca::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "eigenvalues", "eigenvectors", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpeigenvalues = (obj_)->get(algorithms::pca::eigenvalues);
    SET_ELT(res_, 0, native_type(tmpeigenvalues, gc), names);
    daal::data_management::NumericTablePtr tmpeigenvectors = (obj_)->get(algorithms::pca::eigenvectors);
    SET_ELT(res_, 1, native_type(tmpeigenvectors, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::pca::Result >(obj_), daal::services::SharedPtr< daal::algorithms::pca::Result >, gc);
    SET_ELT(res_, 2, dp_, names);

    return res_;
}

// *****************************************
// algorithms::kernel_function::linear

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::kernel_function::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "values", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpvalues = (obj_)->get(algorithms::kernel_function::values);
    SET_ELT(res_, 0, native_type(tmpvalues, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::kernel_function::Result >(obj_), daal::services::SharedPtr< daal::algorithms::kernel_function::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::kernel_function

// *****************************************
// algorithms::kernel_function::rbf

// *****************************************
// algorithms::svm

// *****************************************
// algorithms::svm::training

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::svm::training::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "model", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::algorithms::svm::ModelPtr tmpmodel = (obj_)->get(algorithms::classifier::training::model);
    SET_ELT(res_, 0, native_type(tmpmodel, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::svm::training::Result >(obj_), daal::services::SharedPtr< daal::algorithms::svm::training::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::svm::prediction

// *****************************************
// algorithms::linear_regression

// *****************************************
// algorithms::linear_regression::prediction

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::linear_regression::prediction::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "prediction", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpprediction = (obj_)->get(algorithms::linear_regression::prediction::prediction);
    SET_ELT(res_, 0, native_type(tmpprediction, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::linear_regression::prediction::Result >(obj_), daal::services::SharedPtr< daal::algorithms::linear_regression::prediction::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::linear_regression::training

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::linear_regression::training::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "model", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::algorithms::linear_regression::ModelPtr tmpmodel = (obj_)->get(algorithms::linear_regression::training::model);
    SET_ELT(res_, 0, native_type(tmpmodel, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::linear_regression::training::Result >(obj_), daal::services::SharedPtr< daal::algorithms::linear_regression::training::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::univariate_outlier_detection

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::univariate_outlier_detection::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "weights", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpweights = (obj_)->get(algorithms::univariate_outlier_detection::weights);
    SET_ELT(res_, 0, native_type(tmpweights, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::univariate_outlier_detection::Result >(obj_), daal::services::SharedPtr< daal::algorithms::univariate_outlier_detection::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::multivariate_outlier_detection

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::multivariate_outlier_detection::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "weights", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpweights = (obj_)->get(algorithms::multivariate_outlier_detection::weights);
    SET_ELT(res_, 0, native_type(tmpweights, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::multivariate_outlier_detection::Result >(obj_), daal::services::SharedPtr< daal::algorithms::multivariate_outlier_detection::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::svd

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::svd::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "singularValues", "leftSingularMatrix", "rightSingularMatrix", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::data_management::NumericTablePtr tmpsingularValues = (obj_)->get(algorithms::svd::singularValues);
    SET_ELT(res_, 0, native_type(tmpsingularValues, gc), names);
    daal::data_management::NumericTablePtr tmpleftSingularMatrix = (obj_)->get(algorithms::svd::leftSingularMatrix);
    SET_ELT(res_, 1, native_type(tmpleftSingularMatrix, gc), names);
    daal::data_management::NumericTablePtr tmprightSingularMatrix = (obj_)->get(algorithms::svd::rightSingularMatrix);
    SET_ELT(res_, 2, native_type(tmprightSingularMatrix, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::svd::Result >(obj_), daal::services::SharedPtr< daal::algorithms::svd::Result >, gc);
    SET_ELT(res_, 3, dp_, names);

    return res_;
}

// *****************************************
// algorithms::multi_class_classifier::training

NTYPE native_type(daal::services::SharedPtr< daal::algorithms::multi_class_classifier::training::Result > & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = {  "model", "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, NULL, gc); /* list */
    daal::algorithms::multi_class_classifier::ModelPtr tmpmodel = (obj_)->get(algorithms::classifier::training::model);
    SET_ELT(res_, 0, native_type(tmpmodel, gc), names);
    MK_DAALPTR(dp_, new daal::services::SharedPtr< daal::algorithms::multi_class_classifier::training::Result >(obj_), daal::services::SharedPtr< daal::algorithms::multi_class_classifier::training::Result >, gc);
    SET_ELT(res_, 1, dp_, names);

    return res_;
}

// *****************************************
// algorithms::multi_class_classifier

// *****************************************
// algorithms::multi_class_classifier::prediction

// *****************************************
// algorithms::classifier
%}


%{

// *****************************************
// algorithms::multinomial_naive_bayes

// *****************************************
// algorithms::multinomial_naive_bayes::prediction




template<typename fptype, algorithms::multinomial_naive_bayes::prediction::Method method>
struct multinomial_naive_bayes_prediction_manager : public multinomial_naive_bayes_prediction_i
{
    typedef algorithms::multinomial_naive_bayes::prediction::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const size_t _p_nClasses;
    const daal::data_management::NumericTablePtr _p_priorClassEstimates ;
    const daal::data_management::NumericTablePtr _p_alpha ;
    TableOrFList  _i_data;
    daal::algorithms::multinomial_naive_bayes::ModelPtr _i_model;
    const bool _distributed;

    multinomial_naive_bayes_prediction_manager(const size_t p_nClasses,
            const daal::data_management::NumericTablePtr p_priorClassEstimates = data_management::NumericTablePtr(),
            const daal::data_management::NumericTablePtr p_alpha = data_management::NumericTablePtr(),
            bool distributed = false)
        : multinomial_naive_bayes_prediction_i()
        , _p_nClasses(p_nClasses)
        , _p_priorClassEstimates(p_priorClassEstimates)
        , _p_alpha(p_alpha)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_priorClassEstimates)) parameter.priorClassEstimates = _p_priorClassEstimates;
        if(! use_default(_p_alpha)) parameter.alpha = _p_alpha;
    }

    virtual classifier_prediction_Batch_i::classifier_prediction_BatchPtr_type get_classifier_prediction_BatchPtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type(_p_nClasses));
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob(_p_nClasses);
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::classifier::prediction::data, _i_data.table);
        if(_i_model) algob.input.set(algorithms::classifier::prediction::model, _i_model);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::algorithms::multinomial_naive_bayes::ModelPtr i_model)
    {
        _i_data = i_data;
        _i_model = i_model;

        return batch();
    }
};


// *****************************************
// algorithms::multinomial_naive_bayes::training




template<typename fptype, algorithms::multinomial_naive_bayes::training::Method method>
struct multinomial_naive_bayes_training_manager : public multinomial_naive_bayes_training_i
{
    typedef algorithms::multinomial_naive_bayes::training::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const size_t _p_nClasses;
    const daal::data_management::NumericTablePtr _p_priorClassEstimates ;
    const daal::data_management::NumericTablePtr _p_alpha ;
    TableOrFList  _i_data;
    TableOrFList  _i_labels;
    const bool _distributed;

    multinomial_naive_bayes_training_manager(const size_t p_nClasses,
            const daal::data_management::NumericTablePtr p_priorClassEstimates = data_management::NumericTablePtr(),
            const daal::data_management::NumericTablePtr p_alpha = data_management::NumericTablePtr(),
            bool distributed = false)
        : multinomial_naive_bayes_training_i()
        , _p_nClasses(p_nClasses)
        , _p_priorClassEstimates(p_priorClassEstimates)
        , _p_alpha(p_alpha)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_priorClassEstimates)) parameter.priorClassEstimates = _p_priorClassEstimates;
        if(! use_default(_p_alpha)) parameter.alpha = _p_alpha;
    }

    virtual classifier_training_Batch_i::classifier_training_BatchPtr_type get_classifier_training_BatchPtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type(_p_nClasses));
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob(_p_nClasses);
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::classifier::training::data, _i_data.table);
        if(!_i_labels.table && _i_labels.file.size()) _i_labels.table = readCSV(_i_labels.file);
        if(_i_labels.table) algob.input.set(algorithms::classifier::training::labels, _i_labels.table);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

    // Distributed computing
public:
    typedef algorithms::multinomial_naive_bayes::training::Distributed<step1Local, fptype, method> algostep1Local_type;
    typedef PartialIOManager2< algostep1Local_type, data_management::NumericTablePtr, data_management::NumericTablePtr, services::SharedPtr< algorithms::multinomial_naive_bayes::training::PartialResult > > iomstep1Local_type;

    typedef algorithms::multinomial_naive_bayes::training::Distributed<step2Master, fptype, method> algostep2Master_type;
    typedef IOManager< algostep2Master_type, services::SharedPtr< algorithms::multinomial_naive_bayes::training::PartialResult >, algorithms::multinomial_naive_bayes::training::ResultPtr > iomstep2Master_type;


    typename iomstep1Local_type::result_type run_step1Local(const typename iomstep1Local_type::input1_type & input1, const typename iomstep1Local_type::input2_type & input2)
    {
        algostep1Local_type algostep1Local(_p_nClasses);
        init_parameters(algostep1Local.parameter);

        if(input1) algostep1Local.input.set(algorithms::classifier::training::data, input1);
        if(input2) algostep1Local.input.set(algorithms::classifier::training::labels, input2);

        algostep1Local.compute();
        if(iomstep1Local_type::needsFini()) {
            algostep1Local.finalizeCompute();
        }
        return iomstep1Local_type::getResult(algostep1Local);
    }

    typename iomstep2Master_type::result_type run_step2Master(const std::vector< typename iomstep2Master_type::input1_type > & input)
    {
        algostep2Master_type algostep2Master(_p_nClasses);
        init_parameters(algostep2Master.parameter);

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep2Master.input.add(algorithms::multinomial_naive_bayes::training::partialModels, *data);
        }

        algostep2Master.compute();
        if(iomstep2Master_type::needsFini()) {
            algostep2Master.finalizeCompute();
        }
        return iomstep2Master_type::getResult(algostep2Master);
    }


    enum {NI = 2};

private:
    NTYPE distributed()
    {
        typename iomstep2Master_type::result_type daalres = applyGather::applyGather< multinomial_naive_bayes_training_manager< fptype, method > >::compute(_i_data, _i_labels, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_
    multinomial_naive_bayes_training_manager() :
        _i_data(),
        _i_labels(),
        _p_nClasses(),
        _p_priorClassEstimates(),
        _p_alpha()        , _distributed(true)
    {}

    void serialize(CnC::serializer & ser)
    {
        ser
            & _i_labels
            & _p_nClasses
            & _p_priorClassEstimates
            & _p_alpha
;
    }
#endif

public:
    NTYPE compute(const TableOrFList & i_data,
                  const TableOrFList & i_labels)
    {
        _i_data = i_data;
        _i_labels = i_labels;

        return _distributed ? distributed() : batch();
    }
};
#ifdef _DIST_
namespace CnC {
template<typename fptype, algorithms::multinomial_naive_bayes::training::Method method>
    static inline void serialize(serializer & ser, multinomial_naive_bayes_training_manager<fptype, method> *& t)
    {
        ser & chunk< multinomial_naive_bayes_training_manager<fptype, method> >(t, 1);
    }
}
#endif


// *****************************************
// algorithms::kmeans::init




template<typename fptype, algorithms::kmeans::init::Method method>
struct kmeans_init_manager : public kmeans_init_i
{
    typedef algorithms::kmeans::init::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const size_t _p_nClusters;
    const size_t _p_nRowsTotal ;
    const size_t _p_offset ;
    const size_t _p_seed ;
    const double _p_oversamplingFactor ;
    const size_t _p_nRounds ;
    TableOrFList  _i_data;
    const bool _distributed;

    kmeans_init_manager(const size_t p_nClusters,
            const size_t p_nRowsTotal = -1,
            const size_t p_offset = -1,
            const size_t p_seed = -1,
            const double p_oversamplingFactor = std::numeric_limits<double>::quiet_NaN(),
            const size_t p_nRounds = -1,
            bool distributed = false)
        : kmeans_init_i()
        , _p_nClusters(p_nClusters)
        , _p_nRowsTotal(p_nRowsTotal)
        , _p_offset(p_offset)
        , _p_seed(p_seed)
        , _p_oversamplingFactor(p_oversamplingFactor)
        , _p_nRounds(p_nRounds)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_nRowsTotal)) parameter.nRowsTotal = _p_nRowsTotal;
        if(! use_default(_p_offset)) parameter.offset = _p_offset;
        if(! use_default(_p_seed)) parameter.seed = _p_seed;
        if(! use_default(_p_oversamplingFactor)) parameter.oversamplingFactor = _p_oversamplingFactor;
        if(! use_default(_p_nRounds)) parameter.nRounds = _p_nRounds;
    }


    NTYPE batch()
    {
        algob_type algob(_p_nClusters);
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::kmeans::init::data, _i_data.table);

        algob.compute();
        auto daalres = IOManagerSingle< algob_type, services::SharedPtr< typename algob_type::input_type >, data_management::NumericTablePtr, algorithms::kmeans::init::ResultId, algorithms::kmeans::init::centroids >::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

    // Distributed computing
public:
    typedef algorithms::kmeans::init::Distributed<step1Local, fptype, method> algostep1Local_type;
    typedef PartialIOManager< algostep1Local_type, data_management::NumericTablePtr, algorithms::kmeans::init::PartialResultPtr > iomstep1Local_type;

    typedef algorithms::kmeans::init::Distributed<step2Master, fptype, method> algostep2Master_type;
    typedef IOManagerSingle< algostep2Master_type, algorithms::kmeans::init::PartialResultPtr, data_management::NumericTablePtr,algorithms::kmeans::init::ResultId,algorithms::kmeans::init::centroids > iomstep2Master_type;

    typedef algorithms::kmeans::init::Distributed<step2Local, fptype, method> algostep2Local_type;
    typedef PartialIOManager3< algostep2Local_type, data_management::NumericTablePtr, data_management::DataCollectionPtr, data_management::NumericTablePtr, algorithms::kmeans::init::DistributedStep2LocalPlusPlusPartialResultPtr > iomstep2Local_type;

    typedef algorithms::kmeans::init::Distributed<step3Master, fptype, method> algostep3Master_type;
    typedef PartialIOManager< algostep3Master_type, data_management::NumericTablePtr, algorithms::kmeans::init::DistributedStep3MasterPlusPlusPartialResultPtr > iomstep3Master_type;

    typedef algorithms::kmeans::init::Distributed<step4Local, fptype, method> algostep4Local_type;
    typedef PartialIOManager3Single< algostep4Local_type, data_management::NumericTablePtr, data_management::DataCollectionPtr, data_management::NumericTablePtr, data_management::NumericTablePtr,algorithms::kmeans::init::DistributedStep4LocalPlusPlusPartialResultId,algorithms::kmeans::init::outputOfStep4 > iomstep4Local_type;


    typename iomstep1Local_type::result_type run_step1Local(const typename iomstep1Local_type::input1_type & input1, size_t nRowsTotal, size_t offset)
    {
        algostep1Local_type algostep1Local(_p_nClusters, nRowsTotal, offset);
        init_parameters(algostep1Local.parameter);

        if(input1) algostep1Local.input.set(algorithms::kmeans::init::data, input1);

        algostep1Local.compute();
        if(iomstep1Local_type::needsFini()) {
            algostep1Local.finalizeCompute();
        }
        return iomstep1Local_type::getResult(algostep1Local);
    }

    typename iomstep2Master_type::result_type run_step2Master(const std::vector< typename iomstep2Master_type::input1_type > & input)
    {
        algostep2Master_type algostep2Master(_p_nClusters);
        init_parameters(algostep2Master.parameter);

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep2Master.input.add(algorithms::kmeans::init::partialResults, *data);
        }

        algostep2Master.compute();
        if(iomstep2Master_type::needsFini()) {
            algostep2Master.finalizeCompute();
        }
        return iomstep2Master_type::getResult(algostep2Master);
    }

    typename iomstep2Local_type::result_type run_step2Local(const typename iomstep2Local_type::input1_type & input1, const typename iomstep2Local_type::input2_type & input2, const typename iomstep2Local_type::input3_type & input3)
    {
        algostep2Local_type algostep2Local(_p_nClusters, input2 ? false : true);
        init_parameters(algostep2Local.parameter);

        if(input1) algostep2Local.input.set(algorithms::kmeans::init::data, input1);
        if(input2) algostep2Local.input.set(algorithms::kmeans::init::internalInput, input2);
        if(input3) algostep2Local.input.set(algorithms::kmeans::init::inputOfStep2, input3);

        algostep2Local.compute();
        if(iomstep2Local_type::needsFini()) {
            algostep2Local.finalizeCompute();
        }
        return iomstep2Local_type::getResult(algostep2Local);
    }

    typename iomstep3Master_type::result_type run_step3Master(const std::vector< typename iomstep3Master_type::input1_type > & input)
    {
        algostep3Master_type algostep3Master(_p_nClusters);
        init_parameters(algostep3Master.parameter);

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep3Master.input.add(algorithms::kmeans::init::inputOfStep3FromStep2, i, *data);
        }

        algostep3Master.compute();
        if(iomstep3Master_type::needsFini()) {
            algostep3Master.finalizeCompute();
        }
        return iomstep3Master_type::getResult(algostep3Master);
    }

    typename iomstep4Local_type::result_type run_step4Local(const typename iomstep4Local_type::input1_type & input1, const typename iomstep4Local_type::input2_type & input2, const typename iomstep4Local_type::input3_type & input3)
    {
        algostep4Local_type algostep4Local(_p_nClusters);
        init_parameters(algostep4Local.parameter);

        if(input1) algostep4Local.input.set(algorithms::kmeans::init::data, input1);
        if(input2) algostep4Local.input.set(algorithms::kmeans::init::internalInput, input2);
        if(input3) algostep4Local.input.set(algorithms::kmeans::init::inputOfStep4FromStep3, input3);

        algostep4Local.compute();
        if(iomstep4Local_type::needsFini()) {
            algostep4Local.finalizeCompute();
        }
        return iomstep4Local_type::getResult(algostep4Local);
    }


    enum {NI = 1};

private:
    NTYPE distributed()
    {
        typename iomstep4Local_type::result_type daalres = dkmi::dkmi< kmeans_init_manager< fptype, method > >::compute(_i_data, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_
    kmeans_init_manager() :
        _i_data(),
        _p_nClusters(),
        _p_nRowsTotal(),
        _p_offset(),
        _p_seed(),
        _p_oversamplingFactor(),
        _p_nRounds()        , _distributed(true)
    {}

    void serialize(CnC::serializer & ser)
    {
        ser
            & _p_nClusters
            & _p_nRowsTotal
            & _p_offset
            & _p_seed
            & _p_oversamplingFactor
            & _p_nRounds
;
    }
#endif

public:
    NTYPE compute(const TableOrFList & i_data)
    {
        _i_data = i_data;

        return _distributed ? distributed() : batch();
    }
};
#ifdef _DIST_
namespace CnC {
template<typename fptype, algorithms::kmeans::init::Method method>
    static inline void serialize(serializer & ser, kmeans_init_manager<fptype, method> *& t)
    {
        ser & chunk< kmeans_init_manager<fptype, method> >(t, 1);
    }
}
#endif


// *****************************************
// algorithms::kmeans




template<typename fptype, algorithms::kmeans::Method method>
struct kmeans_manager : public kmeans_i
{
    typedef algorithms::kmeans::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const size_t _p_nClusters;
    const size_t _p_maxIterations ;
    const double _p_accuracyThreshold ;
    const double _p_gamma ;
    const std::string _p_assignFlag ;
    TableOrFList  _i_data;
    daal::data_management::NumericTablePtr _i_inputCentroids;
    const bool _distributed;

    kmeans_manager(const size_t p_nClusters,
            const size_t p_maxIterations = -1,
            const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
            const double p_gamma = std::numeric_limits<double>::quiet_NaN(),
            const std::string p_assignFlag = "",
            bool distributed = false)
        : kmeans_i()
        , _p_nClusters(p_nClusters)
        , _p_maxIterations(p_maxIterations)
        , _p_accuracyThreshold(p_accuracyThreshold)
        , _p_gamma(p_gamma)
        , _p_assignFlag(p_assignFlag)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_maxIterations)) parameter.maxIterations = _p_maxIterations;
        if(! use_default(_p_accuracyThreshold)) parameter.accuracyThreshold = _p_accuracyThreshold;
        if(! use_default(_p_gamma)) parameter.gamma = _p_gamma;
        if(! use_default(_p_assignFlag)) parameter.assignFlag = string2bool(_p_assignFlag);
    }


    NTYPE batch()
    {
        algob_type algob(_p_nClusters);
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::kmeans::data, _i_data.table);
        if(_i_inputCentroids) algob.input.set(algorithms::kmeans::inputCentroids, _i_inputCentroids);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

    // Distributed computing
public:
    typedef algorithms::kmeans::Distributed<step1Local, fptype, method> algostep1Local_type;
    typedef PartialIOManager< algostep1Local_type, data_management::NumericTablePtr, algorithms::kmeans::PartialResultPtr > iomstep1Local_type;

    typedef algorithms::kmeans::Distributed<step2Master, fptype, method> algostep2Master_type;
    typedef IOManager< algostep2Master_type, algorithms::kmeans::PartialResultPtr, algorithms::kmeans::ResultPtr > iomstep2Master_type;


    typename iomstep1Local_type::result_type run_step1Local(const typename iomstep1Local_type::input1_type & input1)
    {
        algostep1Local_type algostep1Local(_p_nClusters);
        init_parameters(algostep1Local.parameter);

        if(input1) algostep1Local.input.set(algorithms::kmeans::data, input1);
        if(! use_default(_i_inputCentroids)) algostep1Local.input.set(algorithms::kmeans::inputCentroids,_i_inputCentroids);

        algostep1Local.compute();
        if(iomstep1Local_type::needsFini()) {
            algostep1Local.finalizeCompute();
        }
        return iomstep1Local_type::getResult(algostep1Local);
    }

    typename iomstep2Master_type::result_type run_step2Master(const std::vector< typename iomstep2Master_type::input1_type > & input)
    {
        algostep2Master_type algostep2Master(_p_nClusters);
        init_parameters(algostep2Master.parameter);

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep2Master.input.add(algorithms::kmeans::partialResults, *data);
        }

        algostep2Master.compute();
        if(iomstep2Master_type::needsFini()) {
            algostep2Master.finalizeCompute();
        }
        return iomstep2Master_type::getResult(algostep2Master);
    }


    enum {NI = 1};

private:
    NTYPE distributed()
    {
        typename iomstep2Master_type::result_type daalres = applyGather::applyGather< kmeans_manager< fptype, method > >::compute(_i_data, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_
    kmeans_manager() :
        _i_data(),
        _i_inputCentroids(),
        _p_nClusters(),
        _p_maxIterations(),
        _p_accuracyThreshold(),
        _p_gamma(),
        _p_assignFlag()        , _distributed(true)
    {}

    void serialize(CnC::serializer & ser)
    {
        ser
            & _i_inputCentroids
            & _p_nClusters
            & _p_maxIterations
            & _p_accuracyThreshold
            & _p_gamma
            & _p_assignFlag
;
    }
#endif

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::data_management::NumericTablePtr i_inputCentroids)
    {
        _i_data = i_data;
        _i_inputCentroids = i_inputCentroids;

        return _distributed ? distributed() : batch();
    }
};
#ifdef _DIST_
namespace CnC {
template<typename fptype, algorithms::kmeans::Method method>
    static inline void serialize(serializer & ser, kmeans_manager<fptype, method> *& t)
    {
        ser & chunk< kmeans_manager<fptype, method> >(t, 1);
    }
}
#endif


// *****************************************
// algorithms::pca




template<typename fptype, algorithms::pca::Method method>
struct pca_manager : public pca_i
{
    typedef algorithms::pca::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    TableOrFList  _i_data;
    const bool _distributed;

    pca_manager(bool distributed = false)
        : pca_i()
        , _distributed(distributed)
    {}

private:


    NTYPE batch()
    {
        algob_type algob;

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::pca::data, _i_data.table);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

    // Distributed computing
public:
    typedef algorithms::pca::Distributed<step1Local, fptype, method> algostep1Local_type;
    typedef PartialIOManager< algostep1Local_type, data_management::NumericTablePtr, services::SharedPtr< algorithms::pca::PartialResult< method > > > iomstep1Local_type;

    typedef algorithms::pca::Distributed<step2Master, fptype, method> algostep2Master_type;
    typedef IOManager< algostep2Master_type, services::SharedPtr< algorithms::pca::PartialResult< method > >, algorithms::pca::ResultPtr > iomstep2Master_type;


    typename iomstep1Local_type::result_type run_step1Local(const typename iomstep1Local_type::input1_type & input1)
    {
        algostep1Local_type algostep1Local;

        if(input1) algostep1Local.input.set(algorithms::pca::data, input1);

        algostep1Local.compute();
        if(iomstep1Local_type::needsFini()) {
            algostep1Local.finalizeCompute();
        }
        return iomstep1Local_type::getResult(algostep1Local);
    }

    typename iomstep2Master_type::result_type run_step2Master(const std::vector< typename iomstep2Master_type::input1_type > & input)
    {
        algostep2Master_type algostep2Master;

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep2Master.input.add(algorithms::pca::partialResults, *data);
        }

        algostep2Master.compute();
        if(iomstep2Master_type::needsFini()) {
            algostep2Master.finalizeCompute();
        }
        return iomstep2Master_type::getResult(algostep2Master);
    }


    enum {NI = 1};

private:
    NTYPE distributed()
    {
        typename iomstep2Master_type::result_type daalres = applyGather::applyGather< pca_manager< fptype, method > >::compute(_i_data, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_

    void serialize(CnC::serializer & ser)
    {
        ser
;
    }
#endif

public:
    NTYPE compute(const TableOrFList & i_data)
    {
        _i_data = i_data;

        return _distributed ? distributed() : batch();
    }
};
#ifdef _DIST_
namespace CnC {
template<typename fptype, algorithms::pca::Method method>
    static inline void serialize(serializer & ser, pca_manager<fptype, method> *& t)
    {
        ser & chunk< pca_manager<fptype, method> >(t, 1);
    }
}
#endif


// *****************************************
// algorithms::kernel_function::linear




template<typename fptype, algorithms::kernel_function::linear::Method method>
struct linear_manager : public linear_i
{
    typedef algorithms::kernel_function::linear::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const double _p_k ;
    const double _p_b ;
    const size_t _p_rowIndexX ;
    const size_t _p_rowIndexY ;
    const size_t _p_rowIndexResult ;
    const std::string _p_computationMode ;
    daal::data_management::NumericTablePtr _i_X;
    daal::data_management::NumericTablePtr _i_Y;
    const bool _distributed;

    linear_manager(const double p_k = std::numeric_limits<double>::quiet_NaN(),
            const double p_b = std::numeric_limits<double>::quiet_NaN(),
            const size_t p_rowIndexX = -1,
            const size_t p_rowIndexY = -1,
            const size_t p_rowIndexResult = -1,
            const std::string p_computationMode = "",
            bool distributed = false)
        : linear_i()
        , _p_k(p_k)
        , _p_b(p_b)
        , _p_rowIndexX(p_rowIndexX)
        , _p_rowIndexY(p_rowIndexY)
        , _p_rowIndexResult(p_rowIndexResult)
        , _p_computationMode(p_computationMode)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_k)) parameter.k = _p_k;
        if(! use_default(_p_b)) parameter.b = _p_b;
        if(! use_default(_p_rowIndexX)) parameter.rowIndexX = _p_rowIndexX;
        if(! use_default(_p_rowIndexY)) parameter.rowIndexY = _p_rowIndexY;
        if(! use_default(_p_rowIndexResult)) parameter.rowIndexResult = _p_rowIndexResult;
        if(! use_default(_p_computationMode)) parameter.computationMode = (algorithms::kernel_function::ComputationMode)string2enum_algorithms__kernel_function[_p_computationMode];
    }

    virtual KernelIface_i::KernelIfacePtr_type get_KernelIfacePtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type);
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(_i_X) algob.input.set(algorithms::kernel_function::X, _i_X);
        if(_i_Y) algob.input.set(algorithms::kernel_function::Y, _i_Y);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const daal::data_management::NumericTablePtr i_X,
                  const daal::data_management::NumericTablePtr i_Y)
    {
        _i_X = i_X;
        _i_Y = i_Y;

        return batch();
    }
};


// *****************************************
// algorithms::kernel_function

// *****************************************
// algorithms::kernel_function::rbf




template<typename fptype, algorithms::kernel_function::rbf::Method method>
struct rbf_manager : public rbf_i
{
    typedef algorithms::kernel_function::rbf::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const double _p_sigma ;
    const size_t _p_rowIndexX ;
    const size_t _p_rowIndexY ;
    const size_t _p_rowIndexResult ;
    const std::string _p_computationMode ;
    daal::data_management::NumericTablePtr _i_X;
    daal::data_management::NumericTablePtr _i_Y;
    const bool _distributed;

    rbf_manager(const double p_sigma = std::numeric_limits<double>::quiet_NaN(),
            const size_t p_rowIndexX = -1,
            const size_t p_rowIndexY = -1,
            const size_t p_rowIndexResult = -1,
            const std::string p_computationMode = "",
            bool distributed = false)
        : rbf_i()
        , _p_sigma(p_sigma)
        , _p_rowIndexX(p_rowIndexX)
        , _p_rowIndexY(p_rowIndexY)
        , _p_rowIndexResult(p_rowIndexResult)
        , _p_computationMode(p_computationMode)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_sigma)) parameter.sigma = _p_sigma;
        if(! use_default(_p_rowIndexX)) parameter.rowIndexX = _p_rowIndexX;
        if(! use_default(_p_rowIndexY)) parameter.rowIndexY = _p_rowIndexY;
        if(! use_default(_p_rowIndexResult)) parameter.rowIndexResult = _p_rowIndexResult;
        if(! use_default(_p_computationMode)) parameter.computationMode = (algorithms::kernel_function::ComputationMode)string2enum_algorithms__kernel_function[_p_computationMode];
    }

    virtual KernelIface_i::KernelIfacePtr_type get_KernelIfacePtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type);
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(_i_X) algob.input.set(algorithms::kernel_function::X, _i_X);
        if(_i_Y) algob.input.set(algorithms::kernel_function::Y, _i_Y);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const daal::data_management::NumericTablePtr i_X,
                  const daal::data_management::NumericTablePtr i_Y)
    {
        _i_X = i_X;
        _i_Y = i_Y;

        return batch();
    }
};


// *****************************************
// algorithms::svm

// *****************************************
// algorithms::svm::training




template<typename fptype, algorithms::svm::training::Method method>
struct svm_training_manager : public svm_training_i
{
    typedef algorithms::svm::training::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const double _p_C ;
    const double _p_accuracyThreshold ;
    const double _p_tau ;
    const size_t _p_maxIterations ;
    const size_t _p_cacheSize ;
    const std::string _p_doShrinking ;
    const size_t _p_shrinkingStep ;
    const daal::algorithms::kernel_function::KernelIfacePtr _p_kernel ;
    const size_t _p_nClasses ;
    TableOrFList  _i_data;
    daal::data_management::NumericTablePtr _i_labels;
    const bool _distributed;

    svm_training_manager(const double p_C = std::numeric_limits<double>::quiet_NaN(),
            const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
            const double p_tau = std::numeric_limits<double>::quiet_NaN(),
            const size_t p_maxIterations = -1,
            const size_t p_cacheSize = -1,
            const std::string p_doShrinking = "",
            const size_t p_shrinkingStep = -1,
            const daal::algorithms::kernel_function::KernelIfacePtr p_kernel = daal::algorithms::kernel_function::KernelIfacePtr(),
            const size_t p_nClasses = -1,
            bool distributed = false)
        : svm_training_i()
        , _p_C(p_C)
        , _p_accuracyThreshold(p_accuracyThreshold)
        , _p_tau(p_tau)
        , _p_maxIterations(p_maxIterations)
        , _p_cacheSize(p_cacheSize)
        , _p_doShrinking(p_doShrinking)
        , _p_shrinkingStep(p_shrinkingStep)
        , _p_kernel(p_kernel)
        , _p_nClasses(p_nClasses)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_C)) parameter.C = _p_C;
        if(! use_default(_p_accuracyThreshold)) parameter.accuracyThreshold = _p_accuracyThreshold;
        if(! use_default(_p_tau)) parameter.tau = _p_tau;
        if(! use_default(_p_maxIterations)) parameter.maxIterations = _p_maxIterations;
        if(! use_default(_p_cacheSize)) parameter.cacheSize = _p_cacheSize;
        if(! use_default(_p_doShrinking)) parameter.doShrinking = string2bool(_p_doShrinking);
        if(! use_default(_p_shrinkingStep)) parameter.shrinkingStep = _p_shrinkingStep;
        if(! use_default(_p_kernel)) parameter.kernel = _p_kernel;
        if(! use_default(_p_nClasses)) parameter.nClasses = _p_nClasses;
    }

    virtual classifier_training_Batch_i::classifier_training_BatchPtr_type get_classifier_training_BatchPtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type);
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::classifier::training::data, _i_data.table);
        if(_i_labels) algob.input.set(algorithms::classifier::training::labels, _i_labels);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::data_management::NumericTablePtr i_labels)
    {
        _i_data = i_data;
        _i_labels = i_labels;

        return batch();
    }
};


// *****************************************
// algorithms::svm::prediction




template<typename fptype, algorithms::svm::prediction::Method method>
struct svm_prediction_manager : public svm_prediction_i
{
    typedef algorithms::svm::prediction::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const double _p_C ;
    const double _p_accuracyThreshold ;
    const double _p_tau ;
    const size_t _p_maxIterations ;
    const size_t _p_cacheSize ;
    const std::string _p_doShrinking ;
    const size_t _p_shrinkingStep ;
    const daal::algorithms::kernel_function::KernelIfacePtr _p_kernel ;
    const size_t _p_nClasses ;
    TableOrFList  _i_data;
    daal::algorithms::svm::ModelPtr _i_model;
    const bool _distributed;

    svm_prediction_manager(const double p_C = std::numeric_limits<double>::quiet_NaN(),
            const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
            const double p_tau = std::numeric_limits<double>::quiet_NaN(),
            const size_t p_maxIterations = -1,
            const size_t p_cacheSize = -1,
            const std::string p_doShrinking = "",
            const size_t p_shrinkingStep = -1,
            const daal::algorithms::kernel_function::KernelIfacePtr p_kernel = daal::algorithms::kernel_function::KernelIfacePtr(),
            const size_t p_nClasses = -1,
            bool distributed = false)
        : svm_prediction_i()
        , _p_C(p_C)
        , _p_accuracyThreshold(p_accuracyThreshold)
        , _p_tau(p_tau)
        , _p_maxIterations(p_maxIterations)
        , _p_cacheSize(p_cacheSize)
        , _p_doShrinking(p_doShrinking)
        , _p_shrinkingStep(p_shrinkingStep)
        , _p_kernel(p_kernel)
        , _p_nClasses(p_nClasses)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_C)) parameter.C = _p_C;
        if(! use_default(_p_accuracyThreshold)) parameter.accuracyThreshold = _p_accuracyThreshold;
        if(! use_default(_p_tau)) parameter.tau = _p_tau;
        if(! use_default(_p_maxIterations)) parameter.maxIterations = _p_maxIterations;
        if(! use_default(_p_cacheSize)) parameter.cacheSize = _p_cacheSize;
        if(! use_default(_p_doShrinking)) parameter.doShrinking = string2bool(_p_doShrinking);
        if(! use_default(_p_shrinkingStep)) parameter.shrinkingStep = _p_shrinkingStep;
        if(! use_default(_p_kernel)) parameter.kernel = _p_kernel;
        if(! use_default(_p_nClasses)) parameter.nClasses = _p_nClasses;
    }

    virtual classifier_prediction_Batch_i::classifier_prediction_BatchPtr_type get_classifier_prediction_BatchPtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type);
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::classifier::prediction::data, _i_data.table);
        if(_i_model) algob.input.set(algorithms::classifier::prediction::model, _i_model);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::algorithms::svm::ModelPtr i_model)
    {
        _i_data = i_data;
        _i_model = i_model;

        return batch();
    }
};


// *****************************************
// algorithms::linear_regression

// *****************************************
// algorithms::linear_regression::prediction




template<typename fptype, algorithms::linear_regression::prediction::Method method>
struct linear_regression_prediction_manager : public linear_regression_prediction_i
{
    typedef algorithms::linear_regression::prediction::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    TableOrFList  _i_data;
    daal::algorithms::linear_regression::ModelPtr _i_model;
    const bool _distributed;

    linear_regression_prediction_manager(bool distributed = false)
        : linear_regression_prediction_i()
        , _distributed(distributed)
    {}

private:


    NTYPE batch()
    {
        algob_type algob;

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::linear_regression::prediction::data, _i_data.table);
        if(_i_model) algob.input.set(algorithms::linear_regression::prediction::model, _i_model);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::algorithms::linear_regression::ModelPtr i_model)
    {
        _i_data = i_data;
        _i_model = i_model;

        return batch();
    }
};


// *****************************************
// algorithms::linear_regression::training




template<typename fptype, algorithms::linear_regression::training::Method method>
struct linear_regression_training_manager : public linear_regression_training_i
{
    typedef algorithms::linear_regression::training::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const std::string _p_interceptFlag ;
    TableOrFList  _i_data;
    daal::data_management::NumericTablePtr _i_dependentVariables;
    const bool _distributed;

    linear_regression_training_manager(const std::string p_interceptFlag = "",
            bool distributed = false)
        : linear_regression_training_i()
        , _p_interceptFlag(p_interceptFlag)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_interceptFlag)) parameter.interceptFlag = string2bool(_p_interceptFlag);
    }


    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::linear_regression::training::data, _i_data.table);
        if(_i_dependentVariables) algob.input.set(algorithms::linear_regression::training::dependentVariables, _i_dependentVariables);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

    // Distributed computing
public:
    typedef algorithms::linear_regression::training::Distributed<step1Local, fptype, method> algostep1Local_type;
    typedef PartialIOManager< algostep1Local_type, data_management::NumericTablePtr, services::SharedPtr< algorithms::linear_regression::training::PartialResult > > iomstep1Local_type;

    typedef algorithms::linear_regression::training::Distributed<step2Master, fptype, method> algostep2Master_type;
    typedef IOManager< algostep2Master_type, services::SharedPtr< algorithms::linear_regression::training::PartialResult >, algorithms::linear_regression::training::ResultPtr > iomstep2Master_type;


    typename iomstep1Local_type::result_type run_step1Local(const typename iomstep1Local_type::input1_type & input1)
    {
        algostep1Local_type algostep1Local;
        init_parameters(algostep1Local.parameter);

        if(input1) algostep1Local.input.set(algorithms::linear_regression::training::data, input1);
        if(! use_default(_i_dependentVariables)) algostep1Local.input.set(algorithms::linear_regression::training::dependentVariables,_i_dependentVariables);

        algostep1Local.compute();
        if(iomstep1Local_type::needsFini()) {
            algostep1Local.finalizeCompute();
        }
        return iomstep1Local_type::getResult(algostep1Local);
    }

    typename iomstep2Master_type::result_type run_step2Master(const std::vector< typename iomstep2Master_type::input1_type > & input)
    {
        algostep2Master_type algostep2Master;
        init_parameters(algostep2Master.parameter);

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep2Master.input.add(algorithms::linear_regression::training::partialModels, *data);
        }

        algostep2Master.compute();
        if(iomstep2Master_type::needsFini()) {
            algostep2Master.finalizeCompute();
        }
        return iomstep2Master_type::getResult(algostep2Master);
    }


    enum {NI = 1};

private:
    NTYPE distributed()
    {
        typename iomstep2Master_type::result_type daalres = applyGather::applyGather< linear_regression_training_manager< fptype, method > >::compute(_i_data, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_

    void serialize(CnC::serializer & ser)
    {
        ser
            & _i_dependentVariables
            & _p_interceptFlag
;
    }
#endif

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::data_management::NumericTablePtr i_dependentVariables)
    {
        _i_data = i_data;
        _i_dependentVariables = i_dependentVariables;

        return _distributed ? distributed() : batch();
    }
};
#ifdef _DIST_
namespace CnC {
template<typename fptype, algorithms::linear_regression::training::Method method>
    static inline void serialize(serializer & ser, linear_regression_training_manager<fptype, method> *& t)
    {
        ser & chunk< linear_regression_training_manager<fptype, method> >(t, 1);
    }
}
#endif


// *****************************************
// algorithms::univariate_outlier_detection




template<typename fptype, algorithms::univariate_outlier_detection::Method method>
struct univariate_outlier_detection_manager : public univariate_outlier_detection_i
{
    typedef algorithms::univariate_outlier_detection::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    TableOrFList  _i_data;
    daal::data_management::NumericTablePtr _i_location ;
    daal::data_management::NumericTablePtr _i_scatter ;
    daal::data_management::NumericTablePtr _i_threshold ;
    const bool _distributed;

    univariate_outlier_detection_manager(bool distributed = false)
        : univariate_outlier_detection_i()
        , _distributed(distributed)
    {}

private:


    NTYPE batch()
    {
        algob_type algob;

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::univariate_outlier_detection::data, _i_data.table);
        if(_i_location) algob.input.set(algorithms::univariate_outlier_detection::location, _i_location);
        if(_i_scatter) algob.input.set(algorithms::univariate_outlier_detection::scatter, _i_scatter);
        if(_i_threshold) algob.input.set(algorithms::univariate_outlier_detection::threshold, _i_threshold);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::data_management::NumericTablePtr i_location = data_management::NumericTablePtr(),
                  const daal::data_management::NumericTablePtr i_scatter = data_management::NumericTablePtr(),
                  const daal::data_management::NumericTablePtr i_threshold = data_management::NumericTablePtr())
    {
        _i_data = i_data;
        _i_location = i_location;
        _i_scatter = i_scatter;
        _i_threshold = i_threshold;

        return batch();
    }
};


// *****************************************
// algorithms::multivariate_outlier_detection




template<typename fptype, algorithms::multivariate_outlier_detection::Method method>
struct multivariate_outlier_detection_manager : public multivariate_outlier_detection_i
{
    typedef algorithms::multivariate_outlier_detection::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    TableOrFList  _i_data;
    daal::data_management::NumericTablePtr _i_location ;
    daal::data_management::NumericTablePtr _i_scatter ;
    daal::data_management::NumericTablePtr _i_threshold ;
    const bool _distributed;

    multivariate_outlier_detection_manager(bool distributed = false)
        : multivariate_outlier_detection_i()
        , _distributed(distributed)
    {}

private:


    NTYPE batch()
    {
        algob_type algob;

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::multivariate_outlier_detection::data, _i_data.table);
        if(_i_location) algob.input.set(algorithms::multivariate_outlier_detection::location, _i_location);
        if(_i_scatter) algob.input.set(algorithms::multivariate_outlier_detection::scatter, _i_scatter);
        if(_i_threshold) algob.input.set(algorithms::multivariate_outlier_detection::threshold, _i_threshold);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::data_management::NumericTablePtr i_location = data_management::NumericTablePtr(),
                  const daal::data_management::NumericTablePtr i_scatter = data_management::NumericTablePtr(),
                  const daal::data_management::NumericTablePtr i_threshold = data_management::NumericTablePtr())
    {
        _i_data = i_data;
        _i_location = i_location;
        _i_scatter = i_scatter;
        _i_threshold = i_threshold;

        return batch();
    }
};


// *****************************************
// algorithms::svd




template<typename fptype, algorithms::svd::Method method>
struct svd_manager : public svd_i
{
    typedef algorithms::svd::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const std::string _p_leftSingularMatrix ;
    const std::string _p_rightSingularMatrix ;
    TableOrFList  _i_data;
    const bool _distributed;

    svd_manager(const std::string p_leftSingularMatrix = "",
            const std::string p_rightSingularMatrix = "",
            bool distributed = false)
        : svd_i()
        , _p_leftSingularMatrix(p_leftSingularMatrix)
        , _p_rightSingularMatrix(p_rightSingularMatrix)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_leftSingularMatrix)) parameter.leftSingularMatrix = (algorithms::svd::SVDResultFormat)string2enum_algorithms__svd[_p_leftSingularMatrix];
        if(! use_default(_p_rightSingularMatrix)) parameter.rightSingularMatrix = (algorithms::svd::SVDResultFormat)string2enum_algorithms__svd[_p_rightSingularMatrix];
    }


    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::svd::data, _i_data.table);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

    // Distributed computing
public:
    typedef algorithms::svd::Distributed<step1Local, fptype, method> algostep1Local_type;
    typedef PartialIOManagerSingle< algostep1Local_type, data_management::NumericTablePtr, data_management::DataCollectionPtr,algorithms::svd::PartialResultId,algorithms::svd::outputOfStep1ForStep2 > iomstep1Local_type;

    typedef algorithms::svd::Distributed<step2Master, fptype, method> algostep2Master_type;
    typedef IOManager< algostep2Master_type, data_management::DataCollectionPtr, algorithms::svd::ResultPtr > iomstep2Master_type;


    typename iomstep1Local_type::result_type run_step1Local(const typename iomstep1Local_type::input1_type & input1)
    {
        algostep1Local_type algostep1Local;
        init_parameters(algostep1Local.parameter);

        if(input1) algostep1Local.input.set(algorithms::svd::data, input1);

        algostep1Local.compute();
        if(iomstep1Local_type::needsFini()) {
            algostep1Local.finalizeCompute();
        }
        return iomstep1Local_type::getResult(algostep1Local);
    }

    typename iomstep2Master_type::result_type run_step2Master(const std::vector< typename iomstep2Master_type::input1_type > & input)
    {
        algostep2Master_type algostep2Master;
        init_parameters(algostep2Master.parameter);

        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algostep2Master.input.add(algorithms::svd::inputOfStep2FromStep1, i, *data);
        }

        algostep2Master.compute();
        if(iomstep2Master_type::needsFini()) {
            algostep2Master.finalizeCompute();
        }
        return iomstep2Master_type::getResult(algostep2Master);
    }


    enum {NI = 1};

private:
    NTYPE distributed()
    {
        typename iomstep2Master_type::result_type daalres = applyGather::applyGather< svd_manager< fptype, method > >::compute(_i_data, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_

    void serialize(CnC::serializer & ser)
    {
        ser
            & _p_leftSingularMatrix
            & _p_rightSingularMatrix
;
    }
#endif

public:
    NTYPE compute(const TableOrFList & i_data)
    {
        _i_data = i_data;

        return _distributed ? distributed() : batch();
    }
};
#ifdef _DIST_
namespace CnC {
template<typename fptype, algorithms::svd::Method method>
    static inline void serialize(serializer & ser, svd_manager<fptype, method> *& t)
    {
        ser & chunk< svd_manager<fptype, method> >(t, 1);
    }
}
#endif


// *****************************************
// algorithms::multi_class_classifier::training




template<typename fptype, algorithms::multi_class_classifier::training::Method method>
struct multi_class_classifier_training_manager : public multi_class_classifier_training_i
{
    typedef algorithms::multi_class_classifier::training::Batch<fptype, method> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const size_t _p_maxIterations ;
    const double _p_accuracyThreshold ;
    const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> _p_training ;
    const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> _p_prediction ;
    const size_t _p_nClasses ;
    TableOrFList  _i_data;
    daal::data_management::NumericTablePtr _i_labels;
    const bool _distributed;

    multi_class_classifier_training_manager(const size_t p_maxIterations = -1,
            const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
            const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> p_training = daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>(),
            const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> p_prediction = daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>(),
            const size_t p_nClasses = -1,
            bool distributed = false)
        : multi_class_classifier_training_i()
        , _p_maxIterations(p_maxIterations)
        , _p_accuracyThreshold(p_accuracyThreshold)
        , _p_training(p_training)
        , _p_prediction(p_prediction)
        , _p_nClasses(p_nClasses)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_maxIterations)) parameter.maxIterations = _p_maxIterations;
        if(! use_default(_p_accuracyThreshold)) parameter.accuracyThreshold = _p_accuracyThreshold;
        if(! use_default(_p_training)) parameter.training = _p_training;
        if(! use_default(_p_prediction)) parameter.prediction = _p_prediction;
        if(! use_default(_p_nClasses)) parameter.nClasses = _p_nClasses;
    }

    virtual classifier_training_Batch_i::classifier_training_BatchPtr_type get_classifier_training_BatchPtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type);
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::classifier::training::data, _i_data.table);
        if(_i_labels) algob.input.set(algorithms::classifier::training::labels, _i_labels);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::data_management::NumericTablePtr i_labels)
    {
        _i_data = i_data;
        _i_labels = i_labels;

        return batch();
    }
};


// *****************************************
// algorithms::multi_class_classifier

// *****************************************
// algorithms::multi_class_classifier::prediction




template<typename fptype, algorithms::multi_class_classifier::prediction::Method pmethod, algorithms::multi_class_classifier::training::Method tmethod>
struct multi_class_classifier_prediction_manager : public multi_class_classifier_prediction_i
{
    typedef algorithms::multi_class_classifier::prediction::Batch<fptype, pmethod, tmethod> algob_type;
    typedef IOManager< algob_type, services::SharedPtr< typename algob_type::input_type >, services::SharedPtr< typename algob_type::result_type > > iomb_type;

    const size_t _p_maxIterations ;
    const double _p_accuracyThreshold ;
    const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> _p_training ;
    const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> _p_prediction ;
    const size_t _p_nClasses ;
    TableOrFList  _i_data;
    daal::algorithms::multi_class_classifier::ModelPtr _i_model;
    const bool _distributed;

    multi_class_classifier_prediction_manager(const size_t p_maxIterations = -1,
            const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
            const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> p_training = daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>(),
            const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> p_prediction = daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>(),
            const size_t p_nClasses = -1,
            bool distributed = false)
        : multi_class_classifier_prediction_i()
        , _p_maxIterations(p_maxIterations)
        , _p_accuracyThreshold(p_accuracyThreshold)
        , _p_training(p_training)
        , _p_prediction(p_prediction)
        , _p_nClasses(p_nClasses)
        , _distributed(distributed)
    {}

private:
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
        if(! use_default(_p_maxIterations)) parameter.maxIterations = _p_maxIterations;
        if(! use_default(_p_accuracyThreshold)) parameter.accuracyThreshold = _p_accuracyThreshold;
        if(! use_default(_p_training)) parameter.training = _p_training;
        if(! use_default(_p_prediction)) parameter.prediction = _p_prediction;
        if(! use_default(_p_nClasses)) parameter.nClasses = _p_nClasses;
    }

    virtual classifier_prediction_Batch_i::classifier_prediction_BatchPtr_type get_classifier_prediction_BatchPtr()
    {
        services::SharedPtr< algob_type > algob(new algob_type);
        init_parameters(algob->parameter);

        return algob;
    }

    NTYPE batch()
    {
        algob_type algob;
        init_parameters(algob.parameter);

        if(!_i_data.table && _i_data.file.size()) _i_data.table = readCSV(_i_data.file);
        if(_i_data.table) algob.input.set(algorithms::classifier::prediction::data, _i_data.table);
        if(_i_model) algob.input.set(algorithms::classifier::prediction::model, _i_model);

        algob.compute();
        auto daalres = iomb_type::getResult(algob);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
    NTYPE compute(const TableOrFList & i_data,
                  const daal::algorithms::multi_class_classifier::ModelPtr i_model)
    {
        _i_data = i_data;
        _i_model = i_model;

        return batch();
    }
};


// *****************************************
// algorithms::classifier
%}


%inline %{
extern "C" {


daal::services::SharedPtr< multinomial_naive_bayes_prediction_i > multinomial_naive_bayes_prediction(
        const size_t p_nClasses,
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const daal::data_management::NumericTablePtr p_priorClassEstimates = data_management::NumericTablePtr(),
        const daal::data_management::NumericTablePtr p_alpha = data_management::NumericTablePtr(),
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< multinomial_naive_bayes_prediction_i >(new multinomial_naive_bayes_prediction_manager<double, algorithms::multinomial_naive_bayes::prediction::defaultDense >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< multinomial_naive_bayes_prediction_i >(new multinomial_naive_bayes_prediction_manager<double, algorithms::multinomial_naive_bayes::prediction::fastCSR >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< multinomial_naive_bayes_prediction_i >(new multinomial_naive_bayes_prediction_manager<float, algorithms::multinomial_naive_bayes::prediction::defaultDense >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< multinomial_naive_bayes_prediction_i >(new multinomial_naive_bayes_prediction_manager<float, algorithms::multinomial_naive_bayes::prediction::fastCSR >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< multinomial_naive_bayes_prediction_i >();
}


daal::services::SharedPtr< multinomial_naive_bayes_training_i > multinomial_naive_bayes_training(
        const size_t p_nClasses,
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const daal::data_management::NumericTablePtr p_priorClassEstimates = data_management::NumericTablePtr(),
        const daal::data_management::NumericTablePtr p_alpha = data_management::NumericTablePtr(),
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< multinomial_naive_bayes_training_i >(new multinomial_naive_bayes_training_manager<double, algorithms::multinomial_naive_bayes::training::defaultDense >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< multinomial_naive_bayes_training_i >(new multinomial_naive_bayes_training_manager<double, algorithms::multinomial_naive_bayes::training::fastCSR >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< multinomial_naive_bayes_training_i >(new multinomial_naive_bayes_training_manager<float, algorithms::multinomial_naive_bayes::training::defaultDense >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< multinomial_naive_bayes_training_i >(new multinomial_naive_bayes_training_manager<float, algorithms::multinomial_naive_bayes::training::fastCSR >(p_nClasses, p_priorClassEstimates, p_alpha, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< multinomial_naive_bayes_training_i >();
}


daal::services::SharedPtr< kmeans_init_i > kmeans_init(
        const size_t p_nClusters,
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const size_t p_nRowsTotal = -1,
        const size_t p_offset = -1,
        const size_t p_seed = -1,
        const double p_oversamplingFactor = std::numeric_limits<double>::quiet_NaN(),
        const size_t p_nRounds = -1,
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "deterministicDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::deterministicDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::defaultDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "randomDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::randomDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "plusPlusDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::plusPlusDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "parallelPlusDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::parallelPlusDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "deterministicCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::deterministicCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "randomCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::randomCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "plusPlusCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::plusPlusCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "parallelPlusCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<double, algorithms::kmeans::init::parallelPlusCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "deterministicDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::deterministicDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::defaultDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "randomDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::randomDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "plusPlusDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::plusPlusDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "parallelPlusDense") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::parallelPlusDense >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "deterministicCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::deterministicCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "randomCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::randomCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "plusPlusCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::plusPlusCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }
        else if(t_method == "parallelPlusCSR") {
            return services::SharedPtr< kmeans_init_i >(new kmeans_init_manager<float, algorithms::kmeans::init::parallelPlusCSR >(p_nClusters, p_nRowsTotal, p_offset, p_seed, p_oversamplingFactor, p_nRounds, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< kmeans_init_i >();
}


daal::services::SharedPtr< kmeans_i > kmeans(
        const size_t p_nClusters,
        const std::string & t_fptype = "double",
        const std::string & t_method = "lloydDense",
        const size_t p_maxIterations = -1,
        const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
        const double p_gamma = std::numeric_limits<double>::quiet_NaN(),
        const std::string p_assignFlag = "",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "lloydDense") {
            return services::SharedPtr< kmeans_i >(new kmeans_manager<double, algorithms::kmeans::lloydDense >(p_nClusters, p_maxIterations, p_accuracyThreshold, p_gamma, p_assignFlag, distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< kmeans_i >(new kmeans_manager<double, algorithms::kmeans::defaultDense >(p_nClusters, p_maxIterations, p_accuracyThreshold, p_gamma, p_assignFlag, distributed));
        }
        else if(t_method == "lloydCSR") {
            return services::SharedPtr< kmeans_i >(new kmeans_manager<double, algorithms::kmeans::lloydCSR >(p_nClusters, p_maxIterations, p_accuracyThreshold, p_gamma, p_assignFlag, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "lloydDense") {
            return services::SharedPtr< kmeans_i >(new kmeans_manager<float, algorithms::kmeans::lloydDense >(p_nClusters, p_maxIterations, p_accuracyThreshold, p_gamma, p_assignFlag, distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< kmeans_i >(new kmeans_manager<float, algorithms::kmeans::defaultDense >(p_nClusters, p_maxIterations, p_accuracyThreshold, p_gamma, p_assignFlag, distributed));
        }
        else if(t_method == "lloydCSR") {
            return services::SharedPtr< kmeans_i >(new kmeans_manager<float, algorithms::kmeans::lloydCSR >(p_nClusters, p_maxIterations, p_accuracyThreshold, p_gamma, p_assignFlag, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< kmeans_i >();
}


daal::services::SharedPtr< pca_i > pca(
        const std::string & t_fptype = "double",
        const std::string & t_method = "correlationDense",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "correlationDense") {
            return services::SharedPtr< pca_i >(new pca_manager<double, algorithms::pca::correlationDense >(distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< pca_i >(new pca_manager<double, algorithms::pca::defaultDense >(distributed));
        }
        else if(t_method == "svdDense") {
            return services::SharedPtr< pca_i >(new pca_manager<double, algorithms::pca::svdDense >(distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "correlationDense") {
            return services::SharedPtr< pca_i >(new pca_manager<float, algorithms::pca::correlationDense >(distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< pca_i >(new pca_manager<float, algorithms::pca::defaultDense >(distributed));
        }
        else if(t_method == "svdDense") {
            return services::SharedPtr< pca_i >(new pca_manager<float, algorithms::pca::svdDense >(distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< pca_i >();
}


daal::services::SharedPtr< linear_i > linear(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const double p_k = std::numeric_limits<double>::quiet_NaN(),
        const double p_b = std::numeric_limits<double>::quiet_NaN(),
        const size_t p_rowIndexX = -1,
        const size_t p_rowIndexY = -1,
        const size_t p_rowIndexResult = -1,
        const std::string p_computationMode = "",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< linear_i >(new linear_manager<double, algorithms::kernel_function::linear::defaultDense >(p_k, p_b, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< linear_i >(new linear_manager<double, algorithms::kernel_function::linear::fastCSR >(p_k, p_b, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< linear_i >(new linear_manager<float, algorithms::kernel_function::linear::defaultDense >(p_k, p_b, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< linear_i >(new linear_manager<float, algorithms::kernel_function::linear::fastCSR >(p_k, p_b, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< linear_i >();
}


daal::services::SharedPtr< rbf_i > rbf(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const double p_sigma = std::numeric_limits<double>::quiet_NaN(),
        const size_t p_rowIndexX = -1,
        const size_t p_rowIndexY = -1,
        const size_t p_rowIndexResult = -1,
        const std::string p_computationMode = "",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< rbf_i >(new rbf_manager<double, algorithms::kernel_function::rbf::defaultDense >(p_sigma, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< rbf_i >(new rbf_manager<double, algorithms::kernel_function::rbf::fastCSR >(p_sigma, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< rbf_i >(new rbf_manager<float, algorithms::kernel_function::rbf::defaultDense >(p_sigma, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }
        else if(t_method == "fastCSR") {
            return services::SharedPtr< rbf_i >(new rbf_manager<float, algorithms::kernel_function::rbf::fastCSR >(p_sigma, p_rowIndexX, p_rowIndexY, p_rowIndexResult, p_computationMode, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< rbf_i >();
}


daal::services::SharedPtr< svm_training_i > svm_training(
        const std::string & t_fptype = "double",
        const std::string & t_method = "boser",
        const double p_C = std::numeric_limits<double>::quiet_NaN(),
        const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
        const double p_tau = std::numeric_limits<double>::quiet_NaN(),
        const size_t p_maxIterations = -1,
        const size_t p_cacheSize = -1,
        const std::string p_doShrinking = "",
        const size_t p_shrinkingStep = -1,
        const daal::algorithms::kernel_function::KernelIfacePtr p_kernel = daal::algorithms::kernel_function::KernelIfacePtr(),
        const size_t p_nClasses = -1,
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "boser") {
            return services::SharedPtr< svm_training_i >(new svm_training_manager<double, algorithms::svm::training::boser >(p_C, p_accuracyThreshold, p_tau, p_maxIterations, p_cacheSize, p_doShrinking, p_shrinkingStep, p_kernel, p_nClasses, distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< svm_training_i >(new svm_training_manager<double, algorithms::svm::training::defaultDense >(p_C, p_accuracyThreshold, p_tau, p_maxIterations, p_cacheSize, p_doShrinking, p_shrinkingStep, p_kernel, p_nClasses, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "boser") {
            return services::SharedPtr< svm_training_i >(new svm_training_manager<float, algorithms::svm::training::boser >(p_C, p_accuracyThreshold, p_tau, p_maxIterations, p_cacheSize, p_doShrinking, p_shrinkingStep, p_kernel, p_nClasses, distributed));
        }
        else if(t_method == "defaultDense") {
            return services::SharedPtr< svm_training_i >(new svm_training_manager<float, algorithms::svm::training::defaultDense >(p_C, p_accuracyThreshold, p_tau, p_maxIterations, p_cacheSize, p_doShrinking, p_shrinkingStep, p_kernel, p_nClasses, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< svm_training_i >();
}


daal::services::SharedPtr< svm_prediction_i > svm_prediction(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const double p_C = std::numeric_limits<double>::quiet_NaN(),
        const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
        const double p_tau = std::numeric_limits<double>::quiet_NaN(),
        const size_t p_maxIterations = -1,
        const size_t p_cacheSize = -1,
        const std::string p_doShrinking = "",
        const size_t p_shrinkingStep = -1,
        const daal::algorithms::kernel_function::KernelIfacePtr p_kernel = daal::algorithms::kernel_function::KernelIfacePtr(),
        const size_t p_nClasses = -1,
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else {
            return services::SharedPtr< svm_prediction_i >(new svm_prediction_manager<double, algorithms::svm::prediction::defaultDense >(p_C, p_accuracyThreshold, p_tau, p_maxIterations, p_cacheSize, p_doShrinking, p_shrinkingStep, p_kernel, p_nClasses, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else {
            return services::SharedPtr< svm_prediction_i >(new svm_prediction_manager<float, algorithms::svm::prediction::defaultDense >(p_C, p_accuracyThreshold, p_tau, p_maxIterations, p_cacheSize, p_doShrinking, p_shrinkingStep, p_kernel, p_nClasses, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< svm_prediction_i >();
}


daal::services::SharedPtr< linear_regression_prediction_i > linear_regression_prediction(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else {
            return services::SharedPtr< linear_regression_prediction_i >(new linear_regression_prediction_manager<double, algorithms::linear_regression::prediction::defaultDense >(distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else {
            return services::SharedPtr< linear_regression_prediction_i >(new linear_regression_prediction_manager<float, algorithms::linear_regression::prediction::defaultDense >(distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< linear_regression_prediction_i >();
}


daal::services::SharedPtr< linear_regression_training_i > linear_regression_training(
        const std::string & t_fptype = "double",
        const std::string & t_method = "normEqDense",
        const std::string p_interceptFlag = "",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< linear_regression_training_i >(new linear_regression_training_manager<double, algorithms::linear_regression::training::defaultDense >(p_interceptFlag, distributed));
        }
        else if(t_method == "normEqDense") {
            return services::SharedPtr< linear_regression_training_i >(new linear_regression_training_manager<double, algorithms::linear_regression::training::normEqDense >(p_interceptFlag, distributed));
        }
        else if(t_method == "qrDense") {
            return services::SharedPtr< linear_regression_training_i >(new linear_regression_training_manager<double, algorithms::linear_regression::training::qrDense >(p_interceptFlag, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< linear_regression_training_i >(new linear_regression_training_manager<float, algorithms::linear_regression::training::defaultDense >(p_interceptFlag, distributed));
        }
        else if(t_method == "normEqDense") {
            return services::SharedPtr< linear_regression_training_i >(new linear_regression_training_manager<float, algorithms::linear_regression::training::normEqDense >(p_interceptFlag, distributed));
        }
        else if(t_method == "qrDense") {
            return services::SharedPtr< linear_regression_training_i >(new linear_regression_training_manager<float, algorithms::linear_regression::training::qrDense >(p_interceptFlag, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< linear_regression_training_i >();
}


daal::services::SharedPtr< univariate_outlier_detection_i > univariate_outlier_detection(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else {
            return services::SharedPtr< univariate_outlier_detection_i >(new univariate_outlier_detection_manager<double, algorithms::univariate_outlier_detection::defaultDense >(distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else {
            return services::SharedPtr< univariate_outlier_detection_i >(new univariate_outlier_detection_manager<float, algorithms::univariate_outlier_detection::defaultDense >(distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< univariate_outlier_detection_i >();
}


daal::services::SharedPtr< multivariate_outlier_detection_i > multivariate_outlier_detection(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< multivariate_outlier_detection_i >(new multivariate_outlier_detection_manager<double, algorithms::multivariate_outlier_detection::defaultDense >(distributed));
        }
        else if(t_method == "baconDense") {
            return services::SharedPtr< multivariate_outlier_detection_i >(new multivariate_outlier_detection_manager<double, algorithms::multivariate_outlier_detection::baconDense >(distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_method == "defaultDense") {
            return services::SharedPtr< multivariate_outlier_detection_i >(new multivariate_outlier_detection_manager<float, algorithms::multivariate_outlier_detection::defaultDense >(distributed));
        }
        else if(t_method == "baconDense") {
            return services::SharedPtr< multivariate_outlier_detection_i >(new multivariate_outlier_detection_manager<float, algorithms::multivariate_outlier_detection::baconDense >(distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< multivariate_outlier_detection_i >();
}


daal::services::SharedPtr< svd_i > svd(
        const std::string & t_fptype = "double",
        const std::string & t_method = "defaultDense",
        const std::string p_leftSingularMatrix = "",
        const std::string p_rightSingularMatrix = "",
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else {
            return services::SharedPtr< svd_i >(new svd_manager<double, algorithms::svd::defaultDense >(p_leftSingularMatrix, p_rightSingularMatrix, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else {
            return services::SharedPtr< svd_i >(new svd_manager<float, algorithms::svd::defaultDense >(p_leftSingularMatrix, p_rightSingularMatrix, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< svd_i >();
}


daal::services::SharedPtr< multi_class_classifier_training_i > multi_class_classifier_training(
        const std::string & t_fptype = "double",
        const std::string & t_method = "oneAgainstOne",
        const size_t p_maxIterations = -1,
        const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
        const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> p_training = daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>(),
        const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> p_prediction = daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>(),
        const size_t p_nClasses = -1,
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else {
            return services::SharedPtr< multi_class_classifier_training_i >(new multi_class_classifier_training_manager<double, algorithms::multi_class_classifier::training::oneAgainstOne >(p_maxIterations, p_accuracyThreshold, p_training, p_prediction, p_nClasses, distributed));
        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else {
            return services::SharedPtr< multi_class_classifier_training_i >(new multi_class_classifier_training_manager<float, algorithms::multi_class_classifier::training::oneAgainstOne >(p_maxIterations, p_accuracyThreshold, p_training, p_prediction, p_nClasses, distributed));
        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< multi_class_classifier_training_i >();
}


daal::services::SharedPtr< multi_class_classifier_prediction_i > multi_class_classifier_prediction(
        const std::string & t_fptype = "double",
        const std::string & t_pmethod = "defaultDense",
        const std::string & t_tmethod = "oneAgainstOne>",
        const size_t p_maxIterations = -1,
        const double p_accuracyThreshold = std::numeric_limits<double>::quiet_NaN(),
        const daal::services::SharedPtr<daal::algorithms::classifier::training::Batch> p_training = daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>(),
        const daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch> p_prediction = daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>(),
        const size_t p_nClasses = -1,
        bool distributed = false
    )
{
    if( false ) {;}
    else if(t_fptype == "double") {
        if( false ) {;}
        else if(t_pmethod == "defaultDense") {
            if( false ) {;}
            else {
                return services::SharedPtr< multi_class_classifier_prediction_i >(new multi_class_classifier_prediction_manager<double, algorithms::multi_class_classifier::prediction::defaultDense, algorithms::multi_class_classifier::training::oneAgainstOne >(p_maxIterations, p_accuracyThreshold, p_training, p_prediction, p_nClasses, distributed));
            }

        }
        else if(t_pmethod == "multiClassClassifierWu") {
            if( false ) {;}
            else {
                return services::SharedPtr< multi_class_classifier_prediction_i >(new multi_class_classifier_prediction_manager<double, algorithms::multi_class_classifier::prediction::multiClassClassifierWu, algorithms::multi_class_classifier::training::oneAgainstOne >(p_maxIterations, p_accuracyThreshold, p_training, p_prediction, p_nClasses, distributed));
            }

        }

    }
    else if(t_fptype == "float") {
        if( false ) {;}
        else if(t_pmethod == "defaultDense") {
            if( false ) {;}
            else {
                return services::SharedPtr< multi_class_classifier_prediction_i >(new multi_class_classifier_prediction_manager<float, algorithms::multi_class_classifier::prediction::defaultDense, algorithms::multi_class_classifier::training::oneAgainstOne >(p_maxIterations, p_accuracyThreshold, p_training, p_prediction, p_nClasses, distributed));
            }

        }
        else if(t_pmethod == "multiClassClassifierWu") {
            if( false ) {;}
            else {
                return services::SharedPtr< multi_class_classifier_prediction_i >(new multi_class_classifier_prediction_manager<float, algorithms::multi_class_classifier::prediction::multiClassClassifierWu, algorithms::multi_class_classifier::training::oneAgainstOne >(p_maxIterations, p_accuracyThreshold, p_training, p_prediction, p_nClasses, distributed));
            }

        }

    }

  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< multi_class_classifier_prediction_i >();
}

} // extern "C"
%}



%{
typedef
        CnC::Internal::dist_init init_type;
init_type * initer = NULL;

struct fini
{
    ~fini()
    {
        if(initer) delete initer;
        initer = NULL;
    }
};
static fini _fini;
%}

#define _DIST_
%inline %{
#ifdef _DIST_
extern "C" {

void daalinit(bool spmd = false, int flag = 0)
{
    if(initer) delete initer;
    auto subscriber = [](){
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< multinomial_naive_bayes_training_manager<double, algorithms::multinomial_naive_bayes::training::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< multinomial_naive_bayes_training_manager<double, algorithms::multinomial_naive_bayes::training::fastCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< multinomial_naive_bayes_training_manager<float, algorithms::multinomial_naive_bayes::training::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< multinomial_naive_bayes_training_manager<float, algorithms::multinomial_naive_bayes::training::fastCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::deterministicDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::randomDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::plusPlusDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::parallelPlusDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::deterministicCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::randomCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::plusPlusCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<double, algorithms::kmeans::init::parallelPlusCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::deterministicDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::randomDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::plusPlusDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::parallelPlusDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::deterministicCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::randomCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::plusPlusCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename dkmi::dkmi< kmeans_init_manager<float, algorithms::kmeans::init::parallelPlusCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< kmeans_manager<double, algorithms::kmeans::lloydDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< kmeans_manager<double, algorithms::kmeans::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< kmeans_manager<double, algorithms::kmeans::lloydCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< kmeans_manager<float, algorithms::kmeans::lloydDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< kmeans_manager<float, algorithms::kmeans::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< kmeans_manager<float, algorithms::kmeans::lloydCSR > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< pca_manager<double, algorithms::pca::correlationDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< pca_manager<double, algorithms::pca::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< pca_manager<double, algorithms::pca::svdDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< pca_manager<float, algorithms::pca::correlationDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< pca_manager<float, algorithms::pca::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< pca_manager<float, algorithms::pca::svdDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< linear_regression_training_manager<double, algorithms::linear_regression::training::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< linear_regression_training_manager<double, algorithms::linear_regression::training::normEqDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< linear_regression_training_manager<double, algorithms::linear_regression::training::qrDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< linear_regression_training_manager<float, algorithms::linear_regression::training::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< linear_regression_training_manager<float, algorithms::linear_regression::training::normEqDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< linear_regression_training_manager<float, algorithms::linear_regression::training::qrDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< svd_manager<double, algorithms::svd::defaultDense > >::context_type >();
        CnC::Internal::factory::subscribe< typename applyGather::applyGather< svd_manager<float, algorithms::svd::defaultDense > >::context_type >();
    };
    initer = new init_type(subscriber, flag, spmd);
}

void daalfini()
{
    if(initer) delete initer;
    initer = NULL;
}

size_t num_procs()
{
    return CnC::tuner_base::numProcs();
}

size_t my_procid()
{
    return CnC::tuner_base::myPid();
}

} // extern "C"
#endif //_DIST_
%}


/*

Algorithm:multinomial_naive_bayes_prediction
Name,Type,Default
i_data, TableOrFList ,None
i_model,ModelPtr,None
p_nClasses, size_t,None
t_fptype,string,double
t_method,string,defaultDense
p_priorClassEstimates,NumericTablePtr,NumericTablePtr()
p_alpha,NumericTablePtr,NumericTablePtr()


Algorithm:multinomial_naive_bayes_training
Name,Type,Default
i_data, TableOrFList ,None
i_labels, TableOrFList ,None
p_nClasses, size_t,None
t_fptype,string,double
t_method,string,defaultDense
p_priorClassEstimates,NumericTablePtr,NumericTablePtr()
p_alpha,NumericTablePtr,NumericTablePtr()


Algorithm:kmeans_init
Name,Type,Default
i_data, TableOrFList ,None
p_nClusters, size_t,None
t_fptype,string,double
t_method,string,defaultDense
p_nRowsTotal, size_t, -1
p_offset, size_t, -1
p_seed, size_t, -1
p_oversamplingFactor, double,quiet_NaN()
p_nRounds, size_t, -1


Algorithm:kmeans
Name,Type,Default
i_data, TableOrFList ,None
i_inputCentroids,NumericTablePtr,None
p_nClusters, size_t,None
t_fptype,string,double
t_method,string,lloydDense
p_maxIterations, size_t, -1
p_accuracyThreshold, double,quiet_NaN()
p_gamma, double,quiet_NaN()
p_assignFlag,string, ""


Algorithm:pca
Name,Type,Default
i_data, TableOrFList ,None
t_fptype,string,double
t_method,string,correlationDense


Algorithm:kernel_function_linear
Name,Type,Default
i_X,NumericTablePtr,None
i_Y,NumericTablePtr,None
t_fptype,string,double
t_method,string,defaultDense
p_k, double,quiet_NaN()
p_b, double,quiet_NaN()
p_rowIndexX, size_t, -1
p_rowIndexY, size_t, -1
p_rowIndexResult, size_t, -1
p_computationMode,string, ""


Algorithm:kernel_function_rbf
Name,Type,Default
i_X,NumericTablePtr,None
i_Y,NumericTablePtr,None
t_fptype,string,double
t_method,string,defaultDense
p_sigma, double,quiet_NaN()
p_rowIndexX, size_t, -1
p_rowIndexY, size_t, -1
p_rowIndexResult, size_t, -1
p_computationMode,string, ""


Algorithm:svm_training
Name,Type,Default
i_data, TableOrFList ,None
i_labels,NumericTablePtr,None
t_fptype,string,double
t_method,string,boser
p_C, double,quiet_NaN()
p_accuracyThreshold, double,quiet_NaN()
p_tau, double,quiet_NaN()
p_maxIterations, size_t, -1
p_cacheSize, size_t, -1
p_doShrinking,string, ""
p_shrinkingStep, size_t, -1
p_kernel,KernelIfacePtr,KernelIfacePtr()
p_nClasses, size_t, -1


Algorithm:svm_prediction
Name,Type,Default
i_data, TableOrFList ,None
i_model,ModelPtr,None
t_fptype,string,double
t_method,string,defaultDense
p_C, double,quiet_NaN()
p_accuracyThreshold, double,quiet_NaN()
p_tau, double,quiet_NaN()
p_maxIterations, size_t, -1
p_cacheSize, size_t, -1
p_doShrinking,string, ""
p_shrinkingStep, size_t, -1
p_kernel,KernelIfacePtr,KernelIfacePtr()
p_nClasses, size_t, -1


Algorithm:linear_regression_prediction
Name,Type,Default
i_data, TableOrFList ,None
i_model,ModelPtr,None
t_fptype,string,double
t_method,string,defaultDense


Algorithm:linear_regression_training
Name,Type,Default
i_data, TableOrFList ,None
i_dependentVariables,NumericTablePtr,None
t_fptype,string,double
t_method,string,normEqDense
p_interceptFlag,string, ""


Algorithm:univariate_outlier_detection
Name,Type,Default
i_data, TableOrFList ,None
t_fptype,string,double
t_method,string,defaultDense
i_location,NumericTablePtr,NumericTablePtr()
i_scatter,NumericTablePtr,NumericTablePtr()
i_threshold,NumericTablePtr,NumericTablePtr()


Algorithm:multivariate_outlier_detection
Name,Type,Default
i_data, TableOrFList ,None
t_fptype,string,double
t_method,string,defaultDense
i_location,NumericTablePtr,NumericTablePtr()
i_scatter,NumericTablePtr,NumericTablePtr()
i_threshold,NumericTablePtr,NumericTablePtr()


Algorithm:svd
Name,Type,Default
i_data, TableOrFList ,None
t_fptype,string,double
t_method,string,defaultDense
p_leftSingularMatrix,string, ""
p_rightSingularMatrix,string, ""


Algorithm:multi_class_classifier_training
Name,Type,Default
i_data, TableOrFList ,None
i_labels,NumericTablePtr,None
t_fptype,string,double
t_method,string,oneAgainstOne
p_maxIterations, size_t, -1
p_accuracyThreshold, double,quiet_NaN()
p_training,Batch>,Batch>()
p_prediction,Batch>,Batch>()
p_nClasses, size_t, -1


Algorithm:multi_class_classifier_prediction
Name,Type,Default
i_data, TableOrFList ,None
i_model,ModelPtr,None
t_fptype,string,double
t_pmethod,string,defaultDense
t_tmethod,string,oneAgainstOne>
p_maxIterations, size_t, -1
p_accuracyThreshold, double,quiet_NaN()
p_training,Batch>,Batch>()
p_prediction,Batch>,Batch>()
p_nClasses, size_t, -1



*/

