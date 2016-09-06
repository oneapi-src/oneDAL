/* file: Input.java */
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

/**
 * @brief Contains classes for running the EM for GMM algorithm
 */

package com.intel.daal.algorithms.em_gmm;

import java.io.Serializable;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.em_gmm.init.InitResult;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INPUT"></a>
 * @brief  %Input objects for the EM for GMM algorithm
 */
public class Input extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Input(DaalContext context, long cInput) {
        super(context);
        this.cObject = cInput;
    }

    public Input(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Sets the input object for the EM for GMM algorithm (means or weights)
     * @param id   Identifier of the input object
     * @param val  Object that corresponds to the given identifier
     */
    public void set(InputId id, Serializable val) {
        if (id == InputId.data || id == InputId.inputWeights || id == InputId.inputMeans) {
            cSetInput(cObject, id.getValue(), ((NumericTable) val).getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets the input covariance object for the EM for GMM algorithm
    * @param id   Identifier of the input object
    * @param val  Object that corresponds to the given identifier
    */
    public void set(InputValuesId id, InitResult val) {
        if (id == InputValuesId.inputValues) {
            cSetInputInputValues(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets the collection of input covariances for the EM for GMM algorithm
    * @param id   Identifier of the input object
    * @param val  Object that corresponds to the given identifier
    */
    public void set(InputCovariancesId id, DataCollection val) {
        if (id == InputCovariancesId.inputCovariances) {
            cSetInputCovariances(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the EM for GMM algorithm (means or weights)
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id == InputId.data || id == InputId.inputWeights || id == InputId.inputMeans) {
            return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Returns the input covariance object of the EM for GMM algorithm
    * @param id Identifier of the input object
    * @return   Input object that corresponds to the given identifier
    */
    public DataCollection get(InputCovariancesId id) {
        if (id != InputCovariancesId.inputCovariances) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetInputCovariancesDataCollection(cObject, id.getValue()));
    }

    protected long cObject;

    private native long cInit(long algAddr, int prec, int method, int cmode);

    private native void cSetInput(long cInput, int id, long ntAddr);

    private native void cSetInputInputValues(long cInput, int id, long ntAddr);

    private native void cSetInputCovariances(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

    private native long cGetInputCovariancesDataCollection(long cInput, int id);
}
