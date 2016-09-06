/* file: PartialResult.java */
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

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the correlation
 *        or variance-covariance matrix algorithm in the online or distributed processing mode
 */
public class PartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Default constructor. Constructs an empty PartialResult object
     * @param context  Context to manage the correlation or variance-covariance matrix algorithm
    */
    public PartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public PartialResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    public PartialResult(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode,
            ComputeStep stage) {
        super(context);
        this.cObject = cGetPartialResult(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(),
                stage.getValue());
    }

    public PartialResult(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode) {
        super(context);
        this.cObject = cGetPartialResult(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(),
                ComputeStep.step1Local.getValue());
    }

    /**
     * Returns the partial result of the correlation or variance-covariance matrix algorithm
     * @param  id   Identifier of the partial result, @ref PartialResultId
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(PartialResultId id) {
        int idValue = id.getValue();
        if (idValue != PartialResultId.nObservations.getValue() && idValue != PartialResultId.crossProduct.getValue()
                && idValue != PartialResultId.sum.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetPartialResultTable(getCObject(), idValue));
    }

    /**
     * Sets the partial result of the correlation or variance-covariance matrix algorithm
     * @param id                 Identifier of the partial result
     * @param value              Object to store the partial result
     */
    public void set(PartialResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (idValue != PartialResultId.nObservations.getValue() && idValue != PartialResultId.crossProduct.getValue()
                && idValue != PartialResultId.sum.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultTable(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResult(long cAlgorithm, int prec, int method, int cmode, int stage);

    private native long cGetPartialResultTable(long cPartialResult, int id);

    private native void cSetPartialResultTable(long cPartialResult, int id, long cNumericTable);
}
