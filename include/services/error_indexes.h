/* file: error_indexes.h */
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
//  Details of errors in Intel(R) DAAL.
//--
*/

#ifndef __ERROR_INDEXES__
#define __ERROR_INDEXES__

namespace daal
{
namespace services
{
/**
 * @defgroup error_handling Handling Errors
 * \brief Contains classes and methods to handle exceptions or errors that can occur during library operation.
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-ENUM-SERVICES__ERRORDETAILID"></a>
 * Available error detail ID to represent additional information in error description
 */
enum ErrorDetailID
{
    NoErrorMessageDetailFound = 0, /*!< No error message detail found */
    Row = 1,                       /*!< Detail about row */
    Column = 2,                    /*!< Detail about column */
    Rank = 3,                      /*!< Detail about rank */
    StatisticsName = 4,            /*!< Detail about statistics name function */
    Method = 5,                    /*!< Detail about method */
    Iteration = 6,                 /*!< Detail about iteration number */
    Component = 7,                 /*!< Detail about component number */
    Minor = 8,                     /*!< Detail about order of matrix minor */
    ArgumentName = 9,              /*!< Detail about argument name */
    ElementInCollection = 10,      /*!< Detail about element in collection */
    Dimension = 11,                /*!< Detail about tensor dimension */
    ParameterName = 12,            /*!< Detail about parameter name */
    OptionalInput = 13,            /*!< Detail about optional input name */
    OptionalResult = 14            /*!< Detail about optional result name */
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORID"></a>
 * Execution statuses
 */
enum ErrorID
{
    // Input errors: -1..-1999
    ErrorMethodNotSupported = -1,                                       /*!< Method not supported by the algorithm */
    ErrorIncorrectNumberOfFeatures = -3,                                /*!< Number of columns in numeric table is incorrect */
    ErrorIncorrectNumberOfObservations = -4,                            /*!< Number of rows in numeric table is incorrect */
    ErrorIncorrectSizeOfArray = -7,                                     /*!< Incorrect size of array */
    ErrorNullParameterNotSupported = -8,                                /*!< Null parameter is not supported by the algorithm */
    ErrorIncorrectNumberOfArguments = -9,                               /*!< Number of arguments is incorrect */
    ErrorIncorrectInputNumericTable = -10,                              /*!< Input numeric table is incorrect */
    ErrorEmptyInputNumericTable = -11,                                  /*!< Input numeric table is empty */
    ErrorIncorrectDataRange = -12,                                      /*!< Data range is incorrect */
    ErrorPrecomputedStatisticsIndexOutOfRange = -13,                    /*!< Precomputed statistics index is out of range */
    ErrorIncorrectNumberOfInputNumericTables = -14,                     /*!< Incorrect number of input numeric tables */
    ErrorIncorrectNumberOfOutputNumericTables = -15,                    /*!< Incorrect number of output numeric tables */
    ErrorNullInputNumericTable = -16,                                   /*!< Null input numeric table is not supported */
    ErrorNullOutputNumericTable = -17,                                  /*!< Null output numeric table is not supported */
    ErrorNullModel = -18,                                               /*!< Null model is not supported */
    ErrorInconsistentNumberOfRows = -19,                                /*!< Number of rows in provided numeric tables is inconsistent */
    ErrorIncorrectSizeOfInputNumericTable = -20,                        /*!< Number of columns or rows in input numeric table is incorrect */
    ErrorIncorrectSizeOfOutputNumericTable = -21,                       /*!< Number of columns or rows in output numeric table is incorrect */
    ErrorIncorrectNumberOfRowsInInputNumericTable = -24,                /*!< Number of rows in input numeric table is incorrect */
    ErrorIncorrectNumberOfColumnsInInputNumericTable = -25,             /*!< Number of columns in input numeric table is incorrect */
    ErrorIncorrectNumberOfRowsInOutputNumericTable = -26,               /*!< Number of rows in output numeric table is incorrect */
    ErrorIncorrectNumberOfColumnsInOutputNumericTable = -27,            /*!< Number of columns in output numeric table is incorrect */
    ErrorIncorrectTypeOfInputNumericTable = -28,                        /*!< Incorrect type of input NumericTable */
    ErrorIncorrectTypeOfOutputNumericTable = -29,                       /*!< Incorrect type of output NumericTable */
    ErrorIncorrectNumberOfElementsInInputCollection = -30,              /*!< Incorrect number of elements in input collection */
    ErrorIncorrectNumberOfElementsInResultCollection = -31,             /*!< Incorrect number of elements in result collection */
    ErrorNullInput = -32,                                               /*!< Input not set */
    ErrorNullResult = -33,                                              /*!< Result not set */
    ErrorIncorrectParameter = -34,                                      /*!< Incorrect parameter */
    ErrorModelNotFullInitialized = -35,                                 /*!< Model is not full initialized */
    ErrorInconsistentNumberOfColumns = -36,                             /*!< Inconsistent number of rows in Numeric Table*/
    ErrorIncorrectIndex = -37,                                          /*!< Index in collection is out of range */
    ErrorDataArchiveInternal = -38,                                     /*!< Incorrect size of data block */
    ErrorNullPartialModel = -39,                                        /*!< Null partial model is not supported */
    ErrorNullInputDataCollection = -40,                                 /*!< Null input data collection is not supported */
    ErrorNullOutputDataCollection = -41,                                /*!< Null output data collection is not supported */
    ErrorNullPartialResult = -42,                                       /*!< Partial result not set */
    ErrorIncorrectNumberOfInputNumericTensors = -43,                    /*!< Incorrect number of elements in input collection */
    ErrorIncorrectNumberOfOutputNumericTensors = -44,                   /*!< Incorrect number of elements in output collection */
    ErrorNullTensor = -45,                                              /*!< Null tensor is not supported */
    ErrorIncorrectNumberOfDimensionsInTensor = -46,                     /*!< Number of dimensions in tensor is incorrect */
    ErrorIncorrectSizeOfDimensionInTensor = -47,                        /*!< Size of the dimension in input tensor is incorrect */
    ErrorNullLayerData = -48,                                           /*!< Null layer data is not supported */
    ErrorIncorrectSizeOfLayerData = -49,                                /*!< Incorrect number of elements in layer data collection */
    ErrorNullNumericTable = -50,                                        /*!< Null numeric table is not supported */
    ErrorIncorrectNumberOfColumns = -51,                                /*!< Number of columns in numeric table is incorrect */
    ErrorIncorrectNumberOfRows = -52,                                   /*!< Number of rows in numeric table is incorrect */
    ErrorIncorrectTypeOfNumericTable = -53,                             /*!< Incorrect type of Numeric Table */
    ErrorUnsupportedCSRIndexing = -54,                                  /*!< CSR Numeric Table has unsupported indexing type */
    ErrorSignificanceLevel = -55,                                       /*!< Incorrect significance level value */
    ErrorAccuracyThreshold = -56,                                       /*!< Incorrect accuracy threshold */
    ErrorIncorrectNumberOfBetas = -57,                                  /*!< Incorrect number of betas in linear regression model */
    ErrorIncorrectNumberOfBetasInReducedModel = -58,                    /*!< Incorrect number of betas in reduced linear regression model */
    ErrorNumericTableIsNotSquare = -59,                                 /*!< Numeric table is not square */
    ErrorNullAuxiliaryAlgorithm = -60,                                  /*!< Null auxiliary algorithm */
    ErrorNullInitializationProcedure = -61,                             /*!< Null initialization procedure */
    ErrorNullAuxiliaryDataCollection = -62,                             /*!< Null auxiliary data collection */
    ErrorEmptyAuxiliaryDataCollection = -63,                            /*!< Empty auxiliary data collection */
    ErrorIncorrectElementInCollection = -64,                            /*!< Incorrect element in collection */
    ErrorNullPartialResultDataCollection = -65,                         /*!< Null partial result data collection */
    ErrorIncorrectElementInPartialResultCollection  = -66,              /*!< Incorrect element in collection of partial results */
    ErrorIncorrectElementInNumericTableCollection   = -67,              /*!< Incorrect element in collection of numeric tables */
    ErrorNullOptionalResult = -68,                                      /*!< Null optional result */
    ErrorIncorrectOptionalResult = -69,                                 /*!< Incorrect optional result */
    ErrorIncorrectOptionalInput = -70,                                  /*!< Incorrect optional input */

    // Environment errors: -2000..-2999
    ErrorCpuNotSupported = -2000,                                       /*!< CPU not supported */
    ErrorMemoryAllocationFailed = -2001,                                /*!< Memory allocation failed */
    ErrorEmptyDataBlock = -2004,                                        /*!< Empty data block */

    // Workflow errors: -3000..-3999
    ErrorIncorrectCombinationOfComputationModeAndStep = -3002,          /*!< Incorrect combination of computation mode and computation step */
    ErrorDictionaryAlreadyAvailable = -3003,                            /*!< Data Dictionary is already available */
    ErrorDictionaryNotAvailable = -3004,                                /*!< Data Dictionary is not available */
    ErrorNumericTableNotAvailable = -3005,                              /*!< Numeric Table is not available */
    ErrorNumericTableAlreadyAllocated = -3006,                          /*!< Numeric Table was already allocated */
    ErrorNumericTableNotAllocated = -3007,                              /*!< Numeric Table is not allocated */
    ErrorPrecomputedSumNotAvailable = -3008,                            /*!< Precomputed sums are not available */
    ErrorPrecomputedMinNotAvailable = -3009,                            /*!< Precomputed minimum values are not available */
    ErrorPrecomputedMaxNotAvailable = -3010,                            /*!< Precomputed maximum values are not available */
    ErrorServiceMicroTableInternal = -3011,                             /*!< Numeric Table internal error */
    ErrorEmptyCSRNumericTable = -3012,                                  /*!< CSR Numeric Table is empty */
    ErrorEmptyHomogenNumericTable = -3013,                              /*!< Homogeneous Numeric Table is empty */
    ErrorSourceDataNotAvailable = -3014,                                /*!< Source data is not available */
    ErrorEmptyDataSource = -3015,                                       /*!< Data source is empty */
    ErrorIncorrectClassLabels = -3016,                                  /*!< Class labels provided to classification algorithm are incorrect */
    ErrorIncorrectSizeOfModel = -3017,                                  /*!< Incorrect size of model */
    ErrorIncorrectTypeOfModel = -3018,                                  /*!< Incorrect type of model */

    // Common computation errors: -4000...
    ErrorInputSigmaMatrixHasNonPositiveMinor = -4001,                   /*!< Input sigma matrix has non positive minor */
    ErrorInputSigmaMatrixHasIllegalValue = -4002,                       /*!< Input sigma matrix has illegal value */
    ErrorIncorrectInternalFunctionParameter = -4003,                    /*!< Incorrect parameter in internal function call */

    /* Apriori algorithm errors -5000..-5199 */
    ErrorAprioriIncorrectItemsetTableSize = -5000,                      /*!< Number of rows in the output table containing
                                                                         *   'large' item sets is too small */
    ErrorAprioriIncorrectSupportTableSize = -5001,                      /*!< Number of rows in the output table containing
                                                                         *   'large' item sets support values is too small */
    ErrorAprioriIncorrectLeftRuleTableSize = -5002,                     /*!< Number of rows in the output table containing
                                                                         *   left parts of the association rules is too small */
    ErrorAprioriIncorrectRightRuleTableSize = -5003,                    /*!< Number of rows in the output table containing
                                                                         *   right parts of the association rules is too small */
    ErrorAprioriIncorrectConfidenceTableSize = -5004,                   /*!< Number of rows in the output table containing
                                                                         *   association rules confidence is too small */

    // BrownBoost errors: -5200..-5399

    // Cholesky errors: -5400..-5599
    ErrorCholeskyInternal = -5400,                                      /*!< Cholesky internal error */
    ErrorInputMatrixHasNonPositiveMinor = -5401,                        /*!< Input matrix has non positive minor */

    // Covariance errors: -5600..-5799
    ErrorCovarianceInternal = -5600,                                    /*!< Covariance internal error */

    // Distance errors: -5800..-5999

    // EM errors: -6000..-6099
    ErrorEMMatrixInverse = -6001,                                       /*!< Sigma matrix on M-step cannot be inverted */
    ErrorEMIncorrectToleranceToConverge = -6002,                        /*!< Incorrect value of tolerance to converge in EM parameter */
    ErrorEMIllConditionedCovarianceMatrix = -6003,                      /*!< Ill-conditioned covariance matrix */
    ErrorEMIncorrectMaxNumberOfIterations = -6004,                      /*!< Incorrect maximum number of iterations value in EM parameter */
    ErrorEMNegativeDefinedCovarianceMartix = -6005,                     /*!< Negative-defined covariance matrix */
    ErrorEMEmptyComponent = -6006,                                      /*!< Empty component during computation */
    ErrorEMCovariance = -6007,                                          /*!< Error during covariance computation for component on M step */
    ErrorEMIncorrectNumberOfComponents = -6008,                         /*!< Incorrect number of components value in EM parameter */

    // EM initialization errors: -6100..-6199
    ErrorEMInitNoTrialConverges = -6100,                                /*!< No trial of internal EM start converges */
    ErrorEMInitIncorrectToleranceToConverge = -6101,                    /*!< Incorrect tolerance to converge value in EM initialization parameter */
    ErrorEMInitIncorrectDepthNumberIterations = -6102,                  /*!< Incorrect depth number of iterations value in EM init parameter */
    ErrorEMInitIncorrectNumberOfTrials = -6103,                         /*!< Incorrect number of trials value in EM initialization parameter */
    ErrorEMInitIncorrectNumberOfComponents = -6104,                     /*!< Incorrect numeber of components value in EM initialization parameter */
    ErrorEMInitInconsistentNumberOfComponents = -6105,                  /*!< Inconsistent number of component: number of observations should be
                                                                             greater than number of components */

    // KernelFunction errors: -6200..-6399

    // KMeans errors: -6400..-6599

    // Linear Rergession errors: -6600..-6799
    ErrorLinearRegressionInternal = -6600,                              /*!< Linear Regression internal error */
    ErrorNormEqSystemSolutionFailed = -6601,                            /*!< Failed to solve the system of normal equations */
    ErrorLinRegXtXInvFailed = -6602,                                    /*!< Failed to invert Xt*X matrix */

    // LogitBoots errors: -6800..-6999

    // LowOrderMoments errors: -7000..-7199
    ErrorLowOrderMomentsInternal = -7000,                               /*!< Low Order Moments internal error */

    // MultiClassClassifier errors: -7200..-7399
    ErrorIncorrectNumberOfClasses = -7200,                              /*!< Number of classes provided to classifier is too small */
    ErrorMultiClassNullTwoClassTraining = -7201,                        /*!< Null two-class classifier training algorithm is not supported */
    ErrorMultiClassFailedToTrainTwoClassClassifier = -7202,             /*!< Failed to train a model of two-class classifier */
    ErrorMultiClassFailedToComputeTwoClassPrediction = -7203,           /*!< Failed to compute prediction based on two-class classifier model */

    // NaiveBayes errors: -7400..-7599

    // OutlierDetection errors: -7600..-7799
    ErrorOutlierDetectionInternal = -7600,                              /*!< Outlier Detection internal error */

    /* PCA errors: -7800..-7999 */
    ErrorPCAFailedToComputeCorrelationEigenvalues = -7800,              /*!< Failed to compute eigenvalues of the correlation matrix */
    ErrorPCACorrelationInputDataTypeSupportsOfflineModeOnly = -7801,    /*!< This type of the input data supports
                                                                         *   only offline mode of the computations */
    ErrorIncorrectCrossProductTableSize = -7802,                        /*!< Number of columns or rows
                                                                         *   in cross-product numeric table is incorrect */
    ErrorCrossProductTableIsNotSquare = -7803,                          /*!< Number of columns or rows
                                                                         *   in cross-product numeric table is not equal */
    ErrorInputCorrelationNotSupportedInOnlineAndDistributed = -7804,    /*!< Input correlation matrix is not supported in online and distributed
                                                                             computation modes */

    // QR errors: -8000..-8199

    // Stump errors: -8200..-8399

    ErrorStumpIncorrectSplitFeature = -8200,                            /*!< Incorrect split feature  */

    // SVD errors: -8400..-8599

    // LCN errors: -8400..-8599
    ErrorLCNinnerConvolution = -8400,                                   /*!< Error in convolution 2d layer  */

    // SVM errors: -8600..-8799
    ErrorSVMinnerKernel = -8601,                                        /*!< Error in kernel function */

    // WeakLearner errors: -8800..-8999

    // Compression errors: -9000..-9199
    ErrorCompressionNullInputStream = -9000,                            /*!< Null input stream is not supported */
    ErrorCompressionNullOutputStream = -9001,                           /*!< Null output stream is not supported */

    ErrorCompressionEmptyInputStream = -9002,                           /*!< Input stream of size 0 is not supported */
    ErrorCompressionEmptyOutputStream = -9003,                          /*!< Output stream of size 0 is not supported */

    ErrorZlibInternal = -9004,                                          /*!< Zlib internal error */
    ErrorZlibDataFormat = -9005,                                        /*!< Input compressed stream is in wrong format,
                                                                         *   corrupted or contains not a whole number of compressed blocks */
    ErrorZlibParameters = -9006,                                        /*!< Unsupported Zlib parameters */
    ErrorZlibMemoryAllocationFailed = -9007,                            /*!< Internal Zlib memory allocation failed */
    ErrorZlibNeedDictionary = -9008,                                    /*!< Specific dictionary is needed for decompression,
                                                                         *   currently unsupported Zlib feature */

    ErrorBzip2Internal = -9009,                                         /*!< Bzip2 internal error */
    ErrorBzip2DataFormat = -9010,                                       /*!< Input compressed stream is in wrong format,
                                                                         *   corrupted or contains not a whole number of compressed blocks */
    ErrorBzip2Parameters = -9011,                                       /*!< Unsupported Bzip2 parameters */
    ErrorBzip2MemoryAllocationFailed = -9012,                           /*!< Internal Bzip2 memory allocation failed */

    ErrorLzoInternal = -9013,                                           /*!< LZO internal error */
    ErrorLzoOutputStreamSizeIsNotEnough = -9014,                        /*!< Size of output stream is not enough to start compression */
    ErrorLzoDataFormat = -9015,                                         /*!< Input compressed stream is in wrong format or corrupted */
    ErrorLzoDataFormatLessThenHeader = -9016,                           /*!< Size of input compressed stream is less then
                                                                         *   compressed block header size */
    ErrorLzoDataFormatNotFullBlock = -9017,                             /*!< Input compressed stream contains not a whole
                                                                         *   number of compressed blocks */

    ErrorRleInternal = -9018,                                           /*!< RLE internal error */
    ErrorRleOutputStreamSizeIsNotEnough = -9019,                        /*!< Size of output stream is not enough to start compression */
    ErrorRleDataFormat = -9020,                                         /*!< Input compressed stream is in wrong format or corrupted */
    ErrorRleDataFormatLessThenHeader = -9021,                           /*!< Size of input compressed stream is less then
                                                                         *   compressed block header size */
    ErrorRleDataFormatNotFullBlock = -9022,                             /*!< Input compressed stream contains not a whole
                                                                         *   number of compressed blocks */

    // Quantile error: -10000..-11000
    ErrorQuantileOrderValueIsInvalid = -10001,                          /*!< Quantile order value is invalid */
    ErrorQuantilesInternal = -10002,                                    /*!< Quantile internal error */

    // ALS errors: -11000..-12000
    ErrorALSInternal = -11000,                                          /*!< ALS algorithm internal error */
    ErrorALSInconsistentSparseDataBlocks = -11001,                      /*!< Failed to find a non-zero value with needed indices
                                                                             in a sparse data block */
    // Sorting error: -12000..-13000
    ErrorSortingInternal = -12001,                                      /*!< Sorting internal error */

    // SGD error: -13000..-14000
    ErrorNegativeLearningRate = -13000,                                 /*!< Negative learning rate */

    // Normalization errors: -14000..-15000
    ErrorMeanAndStandardDeviationComputing = -14000,                    /*!< Computation of mean and standard deviation failed */
    ErrorNullVariance = -14001,                                         /*!< Failed to normalize data in column: it has null variance deviation */

    //Sum of functions error: -15000..-16000
    ErrorZeroNumberOfTerms = -15000,                                    /*!< Number of terms can not be zero */

    //Covolution layer error: -16000..-17000
    ErrorConvolutionInternal = -16000,                                  /*!< Convoltion internal error */

    // Ridge Regression errors: -17000..-17999
    ErrorRidgeRegressionInternal = -17000,                              /*!< Ridge Regression internal error */
    ErrorRidgeRegressionNormEqSystemSolutionFailed = -17001,            /*!< Failed to solve the system of normal equations */
    ErrorRidgeRegressionInvertFailed = -17002,                          /*!< Failed to invert matrix */

    // Data management errors:  -80001..
    ErrorUserAllocatedMemory = -80001,                                  /*!< Couldn't free memory allocated by user */

    //Math errors: -90000..-100000
    ErrorDataSourseNotAvailable = -90041,                               /*!< ErrorDataSourseNotAvailable */
    ErrorHandlesSQL = -90042,                                           /*!< ErrorHandlesSQL */
    ErrorODBC = -90043,                                                 /*!< ErrorODBC */
    ErrorSQLstmtHandle = -90044,                                        /*!< ErrorSQLstmtHandle */
    ErrorOnFileOpen = -90045,                                           /*!< Error on file open */

    ErrorKDBNoConnection = -90051,                                      /*!< ErrorKDBNoConnection */
    ErrorKDBWrongCredentials = -90052,                                  /*!< ErrorKDBWrongCredentials */
    ErrorKDBNetworkError = -90053,                                      /*!< ErrorKDBNetworkError */
    ErrorKDBServerError = -90054,                                       /*!< ErrorKDBServerError */
    ErrorKDBTypeUnsupported = -90055,                                   /*!< ErrorKDBTypeUnsupported */
    ErrorKDBWrongTypeOfOutput = -90056,                                 /*!< ErrorKDBWrongTypeOfOutput */

    // Other errors: -100000..
    ErrorObjectDoesNotSupportSerialization = -100000,                   /*!< SerializationIface is not implemented or implemented incorrectly */

    ErrorCouldntAttachCurrentThreadToJavaVM = -110001,                  /*!< Couldn't attach current thread to Java VM */
    ErrorCouldntCreateGlobalReferenceToJavaObject = -110002,            /*!< Couldn't create global reference to Java object */
    ErrorCouldntFindJavaMethod = -110003,                               /*!< Couldn't find Java method */
    ErrorCouldntFindClassForJavaObject = -110004,                       /*!< Couldn't find class for Java object */

    UnknownError = -1000000,                                            /*!< Unknown error */
    NoErrorMessageFound = -1000001                                      /*!< No error message found */
};
/** @} */
}
}

#endif
