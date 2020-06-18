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


#include "oneapi/dal/exception.hpp"
#include "daal/include/services/error_handling.h"
#include "daal/include/services/internal/status_to_error_id.h"

namespace oneapi::dal::backend::interop {

void status_to_exception_default(const daal::services::Status &s) {
    using namespace daal::services;
    using namespace daal::services::internal;

    const ErrorID error = get_error_id(s);
    const char * description = s.getDescription();

    switch (error) {
        case ErrorID::ErrorInconsistentNumberOfRows:
            throw invalid_argument(description);
        case ErrorID::ErrorModelNotFullInitialized:
            throw invalid_argument(description);
        case ErrorID::ErrorInconsistentNumberOfColumns:
            throw invalid_argument(description);
        case ErrorID::ErrorCloneMethodFailed:
            throw invalid_argument(description);
        case ErrorID::ErrorCpuIsInvalid:
            throw invalid_argument(description);
        case ErrorID::ErrorIncorrectCombinationOfComputationModeAndStep:
            throw invalid_argument(description);
        case ErrorID::ErrorInconsistentNumberOfClasses:
            throw invalid_argument(description);
        case ErrorID::ErrorEMInitInconsistentNumberOfComponents:
            throw invalid_argument(description);
        case ErrorID::ErrorNaiveBayesIncorrectModel:
            throw invalid_argument(description);
        case ErrorID::ErrorIncorrectNComponents:
            throw invalid_argument(description);
        case ErrorID::ErrorZlibDataFormat:
            throw invalid_argument(description);
        case ErrorID::ErrorBzip2DataFormat:
            throw invalid_argument(description);
        case ErrorID::ErrorLzoDataFormat:
            throw invalid_argument(description);
        case ErrorID::ErrorQuantileOrderValueIsInvalid:
            throw invalid_argument(description);
        case ErrorID::ErrorALSInconsistentSparseDataBlocks:
            throw invalid_argument(description);
        case ErrorID::ErrorNullVariance:
            throw invalid_argument(description);
        case ErrorID::ErrorDFBootstrapVarImportanceIncompatible:
            throw invalid_argument(description);
        case ErrorID::ErrorDFBootstrapOOBIncompatible:
            throw invalid_argument(description);
        case ErrorID::ErrorGbtIncorrectNumberOfTrees:
            throw invalid_argument(description);
        case ErrorID::ErrorGbtPredictIncorrectNumberOfIterations:
            throw invalid_argument(description);
        case ErrorID::ErrorInconsistenceModelAndBatchSizeInParameter:
            throw invalid_argument(description);
        case ErrorID::ErrorIncorrectNumberOfFeatures:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfObservations:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectSizeOfArray:
            throw domain_error(description);
        case ErrorID::ErrorNullParameterNotSupported:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfArguments:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectInputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorEmptyInputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfInputNumericTables:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfOutputNumericTables:
            throw domain_error(description);
        case ErrorID::ErrorNullInputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorNullOutputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorNullModel:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfColumnsInInputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfRowsInOutputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfColumnsInOutputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectTypeOfInputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectTypeOfOutputNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfElementsInInputCollection:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfElementsInResultCollection:
            throw domain_error(description);
        case ErrorID::ErrorNullInput:
            throw domain_error(description);
        case ErrorID::ErrorNullResult:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectParameter:
            throw domain_error(description);
        case ErrorID::ErrorDataArchiveInternal:
            throw domain_error(description);
        case ErrorID::ErrorNullPartialModel:
            throw domain_error(description);
        case ErrorID::ErrorNullInputDataCollection:
            throw domain_error(description);
        case ErrorID::ErrorNullOutputDataCollection:
            throw domain_error(description);
        case ErrorID::ErrorNullPartialResult:
            throw domain_error(description);
        case ErrorID::ErrorNullLayerData:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectSizeOfLayerData:
            throw domain_error(description);
        case ErrorID::ErrorNullNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfColumns:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfRows:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectTypeOfNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorSignificanceLevel:
            throw domain_error(description);
        case ErrorID::ErrorAccuracyThreshold:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfBetas:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfBetasInReducedModel:
            throw domain_error(description);
        case ErrorID::ErrorNumericTableIsNotSquare:
            throw domain_error(description);
        case ErrorID::ErrorNullAuxiliaryAlgorithm:
            throw domain_error(description);
        case ErrorID::ErrorNullInitializationProcedure:
            throw domain_error(description);
        case ErrorID::ErrorNullAuxiliaryDataCollection:
            throw domain_error(description);
        case ErrorID::ErrorEmptyAuxiliaryDataCollection:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectElementInCollection:
            throw domain_error(description);
        case ErrorID::ErrorNullPartialResultDataCollection:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectElementInPartialResultCollection:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectElementInNumericTableCollection:
            throw domain_error(description);
        case ErrorID::ErrorNullOptionalResult:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectOptionalResult:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectOptionalInput:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfPartialClusters:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectTotalNumberOfPartialClusters:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectDataCollectionSize:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectValueInTheNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectItemInDataCollection:
            throw domain_error(description);
        case ErrorID::ErrorNullPtr:
            throw domain_error(description);
        case ErrorID::ErrorUndefinedFeature:
            throw domain_error(description);
        case ErrorID::ErrorEmptyDataBlock:
            throw domain_error(description);
        case ErrorID::ErrorEmptyHomogenNumericTable:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectSizeOfModel:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectTypeOfModel:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectErrorcodeFromGenerator:
            throw domain_error(description);
        case ErrorID::ErrorInputSigmaMatrixHasNonPositiveMinor:
            throw domain_error(description);
        case ErrorID::ErrorInputSigmaMatrixHasIllegalValue:
            throw domain_error(description);
        case ErrorID::ErrorAprioriIncorrectItemsetTableSize:
            throw domain_error(description);
        case ErrorID::ErrorAprioriIncorrectSupportTableSize:
            throw domain_error(description);
        case ErrorID::ErrorAprioriIncorrectLeftRuleTableSize:
            throw domain_error(description);
        case ErrorID::ErrorAprioriIncorrectRightRuleTableSize:
            throw domain_error(description);
        case ErrorID::ErrorAprioriIncorrectConfidenceTableSize:
            throw domain_error(description);
        case ErrorID::ErrorAprioriIncorrectInputData:
            throw domain_error(description);
        case ErrorID::ErrorInputMatrixHasNonPositiveMinor:
            throw domain_error(description);
        case ErrorID::ErrorEMMatrixInverse:
            throw domain_error(description);
        case ErrorID::ErrorEMIncorrectToleranceToConverge:
            throw domain_error(description);
        case ErrorID::ErrorEMIllConditionedCovarianceMatrix:
            throw domain_error(description);
        case ErrorID::ErrorEMIncorrectMaxNumberOfIterations:
            throw domain_error(description);
        case ErrorID::ErrorEMNegativeDefinedCovarianceMartix:
            throw domain_error(description);
        case ErrorID::ErrorEMIncorrectNumberOfComponents:
            throw domain_error(description);
        case ErrorID::ErrorEMInitIncorrectToleranceToConverge:
            throw domain_error(description);
        case ErrorID::ErrorEMInitIncorrectDepthNumberIterations:
            throw domain_error(description);
        case ErrorID::ErrorEMInitIncorrectNumberOfTrials:
            throw domain_error(description);
        case ErrorID::ErrorEMInitIncorrectNumberOfComponents:
            throw domain_error(description);
        case ErrorID::ErrorKMeansNumberOfClustersIsTooLarge:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfClasses:
            throw domain_error(description);
        case ErrorID::ErrorEmptyInputCollection:
            throw domain_error(description);
        case ErrorID::ErrorPCACorrelationInputDataTypeSupportsOfflineModeOnly:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectCrossProductTableSize:
            throw domain_error(description);
        case ErrorID::ErrorCrossProductTableIsNotSquare:
            throw domain_error(description);
        case ErrorID::ErrorStumpIncorrectSplitFeature:
            throw domain_error(description);
        case ErrorID::ErrorStumpInvalidInputCategoricalData:
            throw domain_error(description);
        case ErrorID::ErrorCompressionNullInputStream:
            throw domain_error(description);
        case ErrorID::ErrorCompressionNullOutputStream:
            throw domain_error(description);
        case ErrorID::ErrorCompressionEmptyInputStream:
            throw domain_error(description);
        case ErrorID::ErrorCompressionEmptyOutputStream:
            throw domain_error(description);
        case ErrorID::ErrorLzoOutputStreamSizeIsNotEnough:
            throw domain_error(description);
        case ErrorID::ErrorLzoDataFormatLessThenHeader:
            throw domain_error(description);
        case ErrorID::ErrorLzoDataFormatNotFullBlock:
            throw domain_error(description);
        case ErrorID::ErrorRleOutputStreamSizeIsNotEnough:
            throw domain_error(description);
        case ErrorID::ErrorRleDataFormat:
            throw domain_error(description);
        case ErrorID::ErrorRleDataFormatLessThenHeader:
            throw domain_error(description);
        case ErrorID::ErrorRleDataFormatNotFullBlock:
            throw domain_error(description);
        case ErrorID::ErrorLowerBoundGreaterThanOrEqualToUpperBound:
            throw domain_error(description);
        case ErrorID::ErrorZeroNumberOfTerms:
            throw domain_error(description);
        case ErrorID::ErrorKDBWrongTypeOfOutput:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectEngineParameter:
            throw domain_error(description);
        case ErrorID::ErrorEmptyInputAlgorithmsCollection:
            throw domain_error(description);
        case ErrorID::ErrorEmptyBuffer:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectOffset:
            throw domain_error(description);
        case ErrorID::ErrorIterativeSolverIncorrectMaxNumberOfIterations:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfTerms:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectNumberOfNodes:
            throw domain_error(description);
        case ErrorID::ErrorIncorrectDataRange:
            throw out_of_range(description);
        case ErrorID::ErrorIncorrectIndex:
            throw out_of_range(description);
        case ErrorID::ErrorMethodNotSupported:
            throw unimplemented_error(description);
        case ErrorID::ErrorUnsupportedCSRIndexing:
            throw unimplemented_error(description);
        case ErrorID::ErrorDataTypeNotSupported:
            throw unimplemented_error(description);
        case ErrorID::ErrorLeapfrogUnsupported:
            throw unimplemented_error(description);
        case ErrorID::ErrorSkipAheadUnsupported:
            throw unimplemented_error(description);
        case ErrorID::ErrorEngineNotSupported:
            throw unimplemented_error(description);
        case ErrorID::ErrorMultiClassNullTwoClassTraining:
            throw unimplemented_error(description);
        case ErrorID::ErrorInputCorrelationNotSupportedInOnlineAndDistributed:
            throw unimplemented_error(description);
        case ErrorID::ErrorZlibParameters:
            throw unimplemented_error(description);
        case ErrorID::ErrorZlibNeedDictionary:
            throw unimplemented_error(description);
        case ErrorID::ErrorBzip2Parameters:
            throw unimplemented_error(description);
        case ErrorID::ErrorKDBTypeUnsupported:
            throw unimplemented_error(description);
        case ErrorID::ErrorObjectDoesNotSupportSerialization:
            throw unimplemented_error(description);
        case ErrorID::ErrorMethodNotImplemented:
            throw unimplemented_error(description);
        case ErrorID::ErrorDeviceSupportNotImplemented:
            throw unimplemented_error(description);
        case ErrorID::ErrorCpuNotSupported:
            throw unavailable_error(description);
        case ErrorID::ErrorDictionaryAlreadyAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorDictionaryNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorNumericTableNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorNumericTableAlreadyAllocated:
            throw unavailable_error(description);
        case ErrorID::ErrorNumericTableNotAllocated:
            throw unavailable_error(description);
        case ErrorID::ErrorPrecomputedSumNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorPrecomputedMinNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorPrecomputedMaxNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorSourceDataNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorFeatureNamesNotAvailable:
            throw unavailable_error(description);
        case ErrorID::ErrorAccessUSMPointerOnOtherDevice:
            throw unavailable_error(description);

        ////////////////////////////////////////////

        case ErrorID::ErrorPCAFailedToComputeCorrelationEigenvalues:
            throw internal_error(description);
        default:
            throw internal_error("Unknown error");
    }
}

template <class local_convertor>
void status_to_exception(const daal::services::Status &s, local_convertor alg_convertor) {
    alg_convertor(s);
    status_to_exception_default(s);
}

} // namespace dal::backend::interop
