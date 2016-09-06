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

package com.intel.daal.algorithms.em_gmm;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__RESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of the
 *        EM for GMM algorithm in the batch processing mode
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

    public Result(DaalContext context, long cAlgorithm, Precision precision, Method method, ComputeMode cmode) {
        super(context);
        cObject = cGetResult(cAlgorithm, precision.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Returns the result of the EM for GMM algorithm (weights or means)
     * @param id   %Result identifier
     * @return         %Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.weights && id != ResultId.means &&
            id != ResultId.nIterations && id != ResultId.goalFunction) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
    * Returns the collection of covariances computed by the EM for GMM algorithm
    * @param id   %Result identifier
    * @return         %Result that corresponds to the given identifier
    */
    public DataCollection get(ResultCovariancesId id) {
        if (id != ResultCovariancesId.covariances) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetCovariancesDataCollection(cObject, id.getValue()));
    }

    /**
     * Returns a covariance with a given index from the collection of computed covariances
     * @param id    Identifier of the collection of covariances
     * @param index Index of the returned covariance
     * @return          Pointer to the covariance table
     */
    public NumericTable get(ResultCovariancesId id, int index) {
        if (id != ResultCovariancesId.covariances) {
            throw new IllegalArgumentException("index arguments for this id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetResultCovarianceTable(cObject, id.getValue(), index));
    }

    /**
     * Sets the result of the EM for GMM algorithm
     * @param id    %Result identifier
     * @param value Numeric table for the result
     */
    public void set(ResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (id != ResultId.weights && id != ResultId.means &&
            id != ResultId.nIterations && id != ResultId.goalFunction) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, idValue, value.getCObject());
    }

    /**
     * Adds the collection of covariance for the EM for GMM algorithm
     * @param id    Identifier of the collection of covariances
     * @param value Collection of covariances
     */
    public void set(ResultCovariancesId id, DataCollection value) {
        int idValue = id.getValue();
        if (id != ResultCovariancesId.covariances) {
            throw new IllegalArgumentException("id unsupported");
        }
        sSetCovarianceCollection(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long cAlgorithm, int precision, int method, int mode);

    private native long cGetResultTable(long cResult, int id);

    private native long cGetCovariancesDataCollection(long cResult, int id);

    private native long cGetResultCovarianceTable(long cResult, int id, int index);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);

    private native void sSetCovarianceCollection(long cResult, int id, long cDataCollection);
}
