/* file: DefaultInitializationProcedure.java */
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

package com.intel.daal.algorithms.covariance;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DEFAULTINITIALIZATIONPROCEDURE"></a>
 * @brief Class that specifies the default method for initialization of partial results for the correlation or variance-covariance matrix algorithm
 */
public class DefaultInitializationProcedure extends InitializationProcedureIface {

    /**
     * Constructs the default initialization procedure
     */
    public DefaultInitializationProcedure() {
        super();
    }

    /**
     * Initializes partial results for the correlation or variance-covariance matrix algorithm
     * @param input         %Input of the algorithm
     * @param partialResult Partial results of the algorithm
     */
    @Override
    public void initialize(Input input, PartialResult partialResult) {

        NumericTable nObservationsTable = partialResult.get(PartialResultId.nObservations);
        NumericTable crossProductTable = partialResult.get(PartialResultId.crossProduct);
        NumericTable sumTable = partialResult.get(PartialResultId.sum);

        long nColumns = sumTable.getNumberOfColumns();
        int iNColumns = (int) nColumns;

        IntBuffer nObservationsBuffer = IntBuffer.allocate(1);
        DoubleBuffer crossProductBuffer = DoubleBuffer.allocate(iNColumns * iNColumns);
        DoubleBuffer sumBuffer = DoubleBuffer.allocate(iNColumns);

        nObservationsBuffer = nObservationsTable.getBlockOfRows(0, 1, nObservationsBuffer);
        crossProductBuffer = crossProductTable.getBlockOfRows(0, nColumns, crossProductBuffer);
        sumBuffer = sumTable.getBlockOfRows(0, 1, sumBuffer);

        nObservationsBuffer.put(0, 0);
        for (int i = 0; i < iNColumns * iNColumns; i++) {
            crossProductBuffer.put(i, 0.0);
        }
        for (int i = 0; i < iNColumns; i++) {
            sumBuffer.put(i, 0.0);
        }

        nObservationsTable.releaseBlockOfRows(0, 1, nObservationsBuffer);
        crossProductTable.releaseBlockOfRows(0, nColumns, crossProductBuffer);
        sumTable.releaseBlockOfRows(0, 1, sumBuffer);
    }
}
