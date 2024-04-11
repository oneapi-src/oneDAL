/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::backend::interop {

void status_to_exception(const daal::services::Status& s) {
    if (s) {
        return;
    }

    using namespace daal::services;
    using namespace daal::services::internal;

    const ErrorID error = get_error_id(s);
    const char* description = s.getDescription();

    switch (error) {
        case ErrorID::ErrorInconsistentNumberOfRows:
        case ErrorID::ErrorModelNotFullInitialized:
        case ErrorID::ErrorInconsistentNumberOfColumns:
        case ErrorID::ErrorCloneMethodFailed:
        case ErrorID::ErrorCpuIsInvalid:
        case ErrorID::ErrorIncorrectCombinationOfComputationModeAndStep:
        case ErrorID::ErrorInconsistentNumberOfClasses:
        case ErrorID::ErrorEMInitInconsistentNumberOfComponents:
        case ErrorID::ErrorNaiveBayesIncorrectModel:
        case ErrorID::ErrorIncorrectNComponents:
        case ErrorID::ErrorZlibDataFormat:
        case ErrorID::ErrorBzip2DataFormat:
        case ErrorID::ErrorLzoDataFormat:
        case ErrorID::ErrorQuantileOrderValueIsInvalid:
        case ErrorID::ErrorALSInconsistentSparseDataBlocks:
        case ErrorID::ErrorNullVariance:
        case ErrorID::ErrorDFBootstrapVarImportanceIncompatible:
        case ErrorID::ErrorDFBootstrapOOBIncompatible:
        case ErrorID::ErrorGbtIncorrectNumberOfTrees:
        case ErrorID::ErrorGbtPredictIncorrectNumberOfIterations:
        case ErrorID::ErrorInconsistenceModelAndBatchSizeInParameter:
            throw invalid_argument(description);
        case ErrorID::ErrorIncorrectNumberOfFeatures:
        case ErrorID::ErrorIncorrectNumberOfObservations:
        case ErrorID::ErrorIncorrectSizeOfArray:
        case ErrorID::ErrorNullParameterNotSupported:
        case ErrorID::ErrorIncorrectNumberOfArguments:
        case ErrorID::ErrorIncorrectInputNumericTable:
        case ErrorID::ErrorEmptyInputNumericTable:
        case ErrorID::ErrorIncorrectNumberOfInputNumericTables:
        case ErrorID::ErrorIncorrectNumberOfOutputNumericTables:
        case ErrorID::ErrorNullInputNumericTable:
        case ErrorID::ErrorNullOutputNumericTable:
        case ErrorID::ErrorNullModel:
        case ErrorID::ErrorIncorrectNumberOfColumnsInInputNumericTable:
        case ErrorID::ErrorIncorrectNumberOfRowsInInputNumericTable:
        case ErrorID::ErrorIncorrectNumberOfRowsInOutputNumericTable:
        case ErrorID::ErrorIncorrectNumberOfColumnsInOutputNumericTable:
        case ErrorID::ErrorIncorrectTypeOfInputNumericTable:
        case ErrorID::ErrorIncorrectTypeOfOutputNumericTable:
        case ErrorID::ErrorIncorrectNumberOfElementsInInputCollection:
        case ErrorID::ErrorIncorrectNumberOfElementsInResultCollection:
        case ErrorID::ErrorNullInput:
        case ErrorID::ErrorNullResult:
        case ErrorID::ErrorIncorrectParameter:
        case ErrorID::ErrorDataArchiveInternal:
        case ErrorID::ErrorNullPartialModel:
        case ErrorID::ErrorNullInputDataCollection:
        case ErrorID::ErrorNullOutputDataCollection:
        case ErrorID::ErrorNullPartialResult:
        case ErrorID::ErrorNullLayerData:
        case ErrorID::ErrorIncorrectSizeOfLayerData:
        case ErrorID::ErrorNullNumericTable:
        case ErrorID::ErrorIncorrectNumberOfColumns:
        case ErrorID::ErrorIncorrectNumberOfRows:
        case ErrorID::ErrorIncorrectTypeOfNumericTable:
        case ErrorID::ErrorSignificanceLevel:
        case ErrorID::ErrorAccuracyThreshold:
        case ErrorID::ErrorIncorrectNumberOfBetas:
        case ErrorID::ErrorIncorrectNumberOfBetasInReducedModel:
        case ErrorID::ErrorNumericTableIsNotSquare:
        case ErrorID::ErrorNullAuxiliaryAlgorithm:
        case ErrorID::ErrorNullInitializationProcedure:
        case ErrorID::ErrorNullAuxiliaryDataCollection:
        case ErrorID::ErrorEmptyAuxiliaryDataCollection:
        case ErrorID::ErrorIncorrectElementInCollection:
        case ErrorID::ErrorNullPartialResultDataCollection:
        case ErrorID::ErrorIncorrectElementInPartialResultCollection:
        case ErrorID::ErrorIncorrectElementInNumericTableCollection:
        case ErrorID::ErrorNullOptionalResult:
        case ErrorID::ErrorIncorrectOptionalResult:
        case ErrorID::ErrorIncorrectOptionalInput:
        case ErrorID::ErrorIncorrectNumberOfPartialClusters:
        case ErrorID::ErrorIncorrectTotalNumberOfPartialClusters:
        case ErrorID::ErrorIncorrectDataCollectionSize:
        case ErrorID::ErrorIncorrectValueInTheNumericTable:
        case ErrorID::ErrorIncorrectItemInDataCollection:
        case ErrorID::ErrorNullPtr:
        case ErrorID::ErrorUndefinedFeature:
        case ErrorID::ErrorEmptyDataBlock:
        case ErrorID::ErrorEmptyHomogenNumericTable:
        case ErrorID::ErrorIncorrectSizeOfModel:
        case ErrorID::ErrorIncorrectTypeOfModel:
        case ErrorID::ErrorInputSigmaMatrixHasNonPositiveMinor:
        case ErrorID::ErrorInputSigmaMatrixHasIllegalValue:
        case ErrorID::ErrorAprioriIncorrectItemsetTableSize:
        case ErrorID::ErrorAprioriIncorrectSupportTableSize:
        case ErrorID::ErrorAprioriIncorrectLeftRuleTableSize:
        case ErrorID::ErrorAprioriIncorrectRightRuleTableSize:
        case ErrorID::ErrorAprioriIncorrectConfidenceTableSize:
        case ErrorID::ErrorAprioriIncorrectInputData:
        case ErrorID::ErrorInputMatrixHasNonPositiveMinor:
        case ErrorID::ErrorEMMatrixInverse:
        case ErrorID::ErrorEMIncorrectToleranceToConverge:
        case ErrorID::ErrorEMIllConditionedCovarianceMatrix:
        case ErrorID::ErrorEMIncorrectMaxNumberOfIterations:
        case ErrorID::ErrorEMNegativeDefinedCovarianceMartix:
        case ErrorID::ErrorEMIncorrectNumberOfComponents:
        case ErrorID::ErrorEMInitIncorrectToleranceToConverge:
        case ErrorID::ErrorEMInitIncorrectDepthNumberIterations:
        case ErrorID::ErrorEMInitIncorrectNumberOfTrials:
        case ErrorID::ErrorEMInitIncorrectNumberOfComponents:
        case ErrorID::ErrorKMeansNumberOfClustersIsTooLarge:
        case ErrorID::ErrorIncorrectNumberOfClasses:
        case ErrorID::ErrorEmptyInputCollection:
        case ErrorID::ErrorPCACorrelationInputDataTypeSupportsOfflineModeOnly:
        case ErrorID::ErrorIncorrectCrossProductTableSize:
        case ErrorID::ErrorCrossProductTableIsNotSquare:
        case ErrorID::ErrorStumpIncorrectSplitFeature:
        case ErrorID::ErrorStumpInvalidInputCategoricalData:
        case ErrorID::ErrorCompressionNullInputStream:
        case ErrorID::ErrorCompressionNullOutputStream:
        case ErrorID::ErrorCompressionEmptyInputStream:
        case ErrorID::ErrorCompressionEmptyOutputStream:
        case ErrorID::ErrorLzoOutputStreamSizeIsNotEnough:
        case ErrorID::ErrorLzoDataFormatLessThenHeader:
        case ErrorID::ErrorLzoDataFormatNotFullBlock:
        case ErrorID::ErrorRleOutputStreamSizeIsNotEnough:
        case ErrorID::ErrorRleDataFormat:
        case ErrorID::ErrorRleDataFormatLessThenHeader:
        case ErrorID::ErrorRleDataFormatNotFullBlock:
        case ErrorID::ErrorLowerBoundGreaterThanOrEqualToUpperBound:
        case ErrorID::ErrorZeroNumberOfTerms:
        case ErrorID::ErrorKDBWrongTypeOfOutput:
        case ErrorID::ErrorIncorrectEngineParameter:
        case ErrorID::ErrorEmptyInputAlgorithmsCollection:
        case ErrorID::ErrorEmptyBuffer:
        case ErrorID::ErrorIncorrectOffset:
        case ErrorID::ErrorIterativeSolverIncorrectMaxNumberOfIterations:
        case ErrorID::ErrorIncorrectNumberOfTerms:
        case ErrorID::ErrorIncorrectNumberOfNodes: throw domain_error(description);
        case ErrorID::ErrorIncorrectDataRange:
        case ErrorID::ErrorIncorrectIndex: throw out_of_range(description);
        case ErrorID::ErrorMethodNotSupported:
        case ErrorID::ErrorUnsupportedCSRIndexing:
        case ErrorID::ErrorDataTypeNotSupported:
        case ErrorID::ErrorLeapfrogUnsupported:
        case ErrorID::ErrorSkipAheadUnsupported:
        case ErrorID::ErrorEngineNotSupported:
        case ErrorID::ErrorMultiClassNullTwoClassTraining:
        case ErrorID::ErrorInputCorrelationNotSupportedInOnlineAndDistributed:
        case ErrorID::ErrorZlibParameters:
        case ErrorID::ErrorZlibNeedDictionary:
        case ErrorID::ErrorBzip2Parameters:
        case ErrorID::ErrorKDBTypeUnsupported:
        case ErrorID::ErrorObjectDoesNotSupportSerialization:
        case ErrorID::ErrorMethodNotImplemented:
        case ErrorID::ErrorDeviceSupportNotImplemented: throw unimplemented(description);
        case ErrorID::ErrorCpuNotSupported:
        case ErrorID::ErrorAccessUSMPointerOnOtherDevice: throw unsupported_device(description);
        case ErrorID::ErrorOnFileOpen:
        case ErrorID::ErrorOnFileRead: throw system_error(std::error_code(), description);
        case ErrorID::ErrorBufferSizeIntegerOverflow:
        case ErrorID::ErrorMemoryCopyFailedInternal:
        case ErrorID::ErrorServiceMicroTableInternal:
        case ErrorID::ErrorIncorrectErrorcodeFromGenerator:
        case ErrorID::ErrorIncorrectInternalFunctionParameter:
        case ErrorID::ErrorUserCancelled:
        case ErrorID::ErrorCholeskyInternal:
        case ErrorID::ErrorCovarianceInternal:
        case ErrorID::ErrorEMCovariance:
        case ErrorID::ErrorVarianceComputation:
        case ErrorID::ErrorLinearRegressionInternal:
        case ErrorID::ErrorNormEqSystemSolutionFailed:
        case ErrorID::ErrorLinRegXtXInvFailed:
        case ErrorID::ErrorLowOrderMomentsInternal:
        case ErrorID::ErrorMultiClassFailedToTrainTwoClassClassifier:
        case ErrorID::ErrorMultiClassFailedToComputeTwoClassPrediction:
        case ErrorID::ErrorOutlierDetectionInternal:
        case ErrorID::ErrorQRInternal:
        case ErrorID::ErrorQrXBDSQRDidNotConverge:
        case ErrorID::ErrorSvdIthParamIllegalValue:
        case ErrorID::ErrorSvdXBDSQRDidNotConverge:
        case ErrorID::ErrorLCNinnerConvolution:
        case ErrorID::ErrorSVMPredictKernerFunctionCall:
        case ErrorID::ErrorZlibInternal:
        case ErrorID::ErrorBzip2Internal:
        case ErrorID::ErrorLzoInternal:
        case ErrorID::ErrorRleInternal:
        case ErrorID::ErrorQuantilesInternal:
        case ErrorID::ErrorALSInternal:
        case ErrorID::ErrorSorting:
        case ErrorID::ErrorMeanAndStandardDeviationComputing:
        case ErrorID::ErrorMinAndMaxComputing:
        case ErrorID::ErrorConvolutionInternal:
        case ErrorID::ErrorRidgeRegressionInternal:
        case ErrorID::ErrorRidgeRegressionNormEqSystemSolutionFailed:
        case ErrorID::ErrorRidgeRegressionInvertFailed:
        case ErrorID::ErrorPivotedQRInternal:
        case ErrorID::ErrorNullByteInjection:
        case ErrorID::ErrorHashTableCollision:
        case ErrorID::UnknownError:
        case ErrorID::NoErrorMessageFound:
        case ErrorID::ErrorEMInitNoTrialConverges:
        case ErrorID::ErrorHandlesSQL:
        case ErrorID::ErrorODBC:
        case ErrorID::ErrorSQLstmtHandle:
        case ErrorID::ErrorCouldntAttachCurrentThreadToJavaVM:
        case ErrorID::ErrorCouldntCreateGlobalReferenceToJavaObject:
        case ErrorID::ErrorCouldntFindJavaMethod:
        case ErrorID::ErrorCouldntFindClassForJavaObject:
        case ErrorID::ErrorCouldntDetachCurrentThreadFromJavaVM:
        case ErrorID::ErrorExecutionContext:
        case ErrorID::ErrorPCAFailedToComputeCorrelationEigenvalues:
        case ErrorID::ErrorDictionaryAlreadyAvailable:
        case ErrorID::ErrorDictionaryNotAvailable:
        case ErrorID::ErrorNumericTableNotAvailable:
        case ErrorID::ErrorNumericTableAlreadyAllocated:
        case ErrorID::ErrorNumericTableNotAllocated:
        case ErrorID::ErrorPrecomputedSumNotAvailable:
        case ErrorID::ErrorPrecomputedMinNotAvailable:
        case ErrorID::ErrorPrecomputedMaxNotAvailable:
        case ErrorID::ErrorSourceDataNotAvailable:
        case ErrorID::ErrorHyperparameterNotFound:
        case ErrorID::ErrorHyperparameterCanNotBeSet:
        case ErrorID::ErrorHyperparameterBadValue:
        case ErrorID::ErrorFeatureNamesNotAvailable: throw internal_error(description);
        case ErrorID::ErrorMemoryAllocationFailed:
        case ErrorID::ErrorZlibMemoryAllocationFailed:
        case ErrorID::ErrorBzip2MemoryAllocationFailed: throw host_bad_alloc();
        default: throw internal_error(dal::detail::error_messages::unknown_status_code());
    }
}

} // namespace oneapi::dal::backend::interop
