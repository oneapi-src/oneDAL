/* file: PartialCorrelationResult.java */
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

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALCORRELATIONRESULT"></a>
 * @brief Provides methods to access partial results obtained with compute() of the %correlation method of PCA algorithm
 *        in the online or distributed processing mode
 */
public final class PartialCorrelationResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public PartialCorrelationResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialCorrelationResult();
    }

    public PartialCorrelationResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    /**
     * Returns partial result of the PCA algorithm
     * @param  id   Identifier of the PartialCorrelationResult, PartialCorrelationResultID
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(PartialCorrelationResultID id) {
        int idValue = id.getValue();
        if (id != PartialCorrelationResultID.nObservations && id != PartialCorrelationResultID.crossProductCorrelation
                && id != PartialCorrelationResultID.sumCorrelation) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetPartialCorrelationResultTable(getCObject(), idValue));
    }

    /**
    * Sets partial result of the PCA algorithm
    * @param id                 Identifier of the partial result
    * @param value              Value of the partial result
    */
    public void set(PartialCorrelationResultID id, NumericTable value) {
        int idValue = id.getValue();
        if (id != PartialCorrelationResultID.nObservations && id != PartialCorrelationResultID.crossProductCorrelation
                && id != PartialCorrelationResultID.sumCorrelation) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialCorrelationResultTable(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialCorrelationResult();

    private native long cGetPartialCorrelationResultTable(long cPartialCorrelationResult, int id);

    private native void cSetPartialCorrelationResultTable(long cPartialCorrelationResult, int id, long cNumericTable);
}
