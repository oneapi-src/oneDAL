/* file: Result.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__RESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of the
 *        correlation or variance-covariance matrix algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    public Result(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode,
                  ComputeStep step) {
        super(context);
        cObject = cGetResult(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(), step.getValue());
    }

    public Result(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode) {
        super(context);
        cObject = cGetResult(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(),
                             ComputeStep.step1Local.getValue());
    }

    /**
     * Returns the result of the correlation or variance-covariance matrix algorithm in the batch processing mode
     * @param id   Identifier of the result, @ref ResultId
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.covariance && id != ResultId.correlation && id != ResultId.mean) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
     * Sets the result of the correlation or variance-covariance matrix algorithm in the batch processing mode
     * @param id   Identifier of the result, @ref ResultId
     * @param val Object to store the result
     */
    public void set(ResultId id, NumericTable val) {
        if (id != ResultId.covariance && id != ResultId.correlation && id != ResultId.mean) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long cAlgorithm, int prec, int method, int mode, int step);

    private native long cGetResultTable(long cObject, int id);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);
}
