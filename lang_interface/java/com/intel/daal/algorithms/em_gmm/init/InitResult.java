/* file: InitResult.java */
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

package com.intel.daal.algorithms.em_gmm.init;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITRESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of InitBatch
 *        for the computation of initial values for the EM for GMM algorithm
 */
public final class InitResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public InitResult(DaalContext context, long cAlgorithm, Precision precision, ComputeMode cmode) {
        super(context);
        cObject = cGetResult(cAlgorithm, precision.getValue(), cmode.getValue());
    }

    /**
     * Returns the initial values for the EM for GMM algorithm
     * @param id   %Result identifier
     * @return         %Result that corresponds to the given identifier
     */
    public NumericTable get(InitResultId id) {
        if (id != InitResultId.weights && id != InitResultId.means) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
    * Returns the collection of initialized covariances for the EM for GMM algorithm
    * @param id   %Result identifier
    * @return         %Result that corresponds to the given identifier
    */
    public DataCollection get(InitResultCovariancesId id) {
        if (id != InitResultCovariancesId.covariances) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetCovariancesDataCollection(cObject, id.getValue()));
    }

    /**
     * Returns the covariance with a given index from the collection of initialized covariances
     * @param id    Identifier of the collection of covariances
     * @param index Index of the covariance to be returned
     * @return          Pointer to the covariance table
     */
    public NumericTable get(InitResultCovariancesId id, int index) {
        if (id != InitResultCovariancesId.covariances) {
            throw new IllegalArgumentException("index arguments for this id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetResultCovarianceTable(cObject, id.getValue(), index));
    }

    /**
     * Sets initial values for the EM for GMM algorithm
     * @param id    %Result identifier
     * @param value Numeric table for the result
     */
    public void set(InitResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (id != InitResultId.weights && id != InitResultId.means) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, idValue, value.getCObject());
    }

    /**
     * Sets the collection of covariances for initialization of the EM for GMM algorithm
     * @param id    Identifier of the collection of covariances
     * @param value Collection of covariances
     */
    public void set(InitResultCovariancesId id, DataCollection value) {
        int idValue = id.getValue();
        if (id != InitResultCovariancesId.covariances) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetCovarianceCollection(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long cAlgorithm, int precision, int mode);

    private native long cGetResultTable(long cResult, int id);

    private native long cGetCovariancesDataCollection(long cResult, int id);

    private native long cGetResultCovarianceTable(long cResult, int id, int index);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);

    private native void cSetCovarianceCollection(long cResult, int id, long cDataCollection);
}
