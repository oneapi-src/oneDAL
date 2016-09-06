/* file: DefaultInitializationProcedureSVD.java */
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

package com.intel.daal.algorithms.pca;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DEFAULTINITIALIZATIONPROCEDURESVD"></a>
 * @brief Class that specifies the default method for partial results initialization
 */
public class DefaultInitializationProcedureSVD extends InitializationProcedureIface {

    /**
     * Constructs default initialization procedure
     */
    public DefaultInitializationProcedureSVD(Method method) {
        super(method);
    }

    /**
     * Initializes partial results
     * @param input         Input objects for the PCA algorithm
     * @param partialResult Partial results of the PCA algorithm
     */
    @Override
    public void initialize(Input input, com.intel.daal.algorithms.PartialResult partialResult) {
        NumericTable dataTable = input.get(InputId.data);
        int nColumns = (int) (dataTable.getNumberOfColumns());

        PartialSVDResult partialSVDResult = (PartialSVDResult) partialResult;

        NumericTable nObservationsSVDTable = partialSVDResult.get(PartialSVDTableResultID.nObservations);
        NumericTable sumSVDTable = partialSVDResult.get(PartialSVDTableResultID.sumSVD);
        NumericTable sumSquaresSVDTable = partialSVDResult.get(PartialSVDTableResultID.sumSquaresSVD);

        IntBuffer nObservationsSVDBuffer = IntBuffer.allocate(1);
        DoubleBuffer sumSVDBuffer = DoubleBuffer.allocate(nColumns);
        DoubleBuffer sumSquaresSVDBuffer = DoubleBuffer.allocate(nColumns);

        nObservationsSVDBuffer = nObservationsSVDTable.getBlockOfRows(0, 1, nObservationsSVDBuffer);
        sumSVDBuffer = sumSVDTable.getBlockOfRows(0, 1, sumSVDBuffer);
        sumSquaresSVDBuffer = sumSquaresSVDTable.getBlockOfRows(0, 1, sumSquaresSVDBuffer);

        nObservationsSVDBuffer.put(0, 0);
        for (int i = 0; i < nColumns; i++) {
            sumSVDBuffer.put(i, 0.0);
            sumSquaresSVDBuffer.put(i, 0.0);
        }

        nObservationsSVDTable.releaseBlockOfRows(0, 1, nObservationsSVDBuffer);
        sumSVDTable.releaseBlockOfRows(0, 1, sumSVDBuffer);
        sumSquaresSVDTable.releaseBlockOfRows(0, 1, sumSquaresSVDBuffer);
    }
}
