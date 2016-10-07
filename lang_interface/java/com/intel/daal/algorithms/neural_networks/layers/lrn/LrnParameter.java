/* file: LrnParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.lrn;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LRNPARAMETER"></a>
 * \brief Class that specifies parameters of the local response normalization layer
 */
public class LrnParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    public LrnParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public LrnParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the numeric table with index of type size_t to calculate local response normalization
     */
    public NumericTable getDimension() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetDimension(cObject));
    }

    /**
     *  Sets the numeric table with index of type size_t to calculate local response normalization
     *  @param dimension   Numeric table of with index of type size_t to calculate local response normalization
     */
    public void setDimension(NumericTable dimension) {
        cSetDimension(cObject, dimension.getCObject());
    }

    /**
     *  Gets the value of hyper-parameter kappa
     */
    public double getKappa() {
        return cGetkappa(cObject);
    }

    /**
     *  Sets the value of hyper-parameter kappa
     *  @param kappa   Value of hyper-parameter kappa
     */
    public void setKappa(double kappa) {
        cSetKappa(cObject, kappa);
    }

    /**
    *  Gets the value of hyper-parameter alpha
    */
    public double getAlpha() {
        return cGetAlpha(cObject);
    }

    /**
     *  Sets the value of hyper-parameter alpha
     *  @param alpha   Value of hyper-parameter alpha
     */
    public void setAlpha(double alpha) {
        cSetAlpha(cObject, alpha);
    }

    /**
    *  Gets the value of hyper-parameter beta
    */
    public double getBeta() {
        return cGetBeta(cObject);
    }

    /**
     *  Sets the value of hyper-parameter beta
     *  @param beta   Value of hyper-parameter beta
     */
    public void setBeta(double beta) {
        cSetBeta(cObject, beta);
    }

    /**
    *  Gets the value of hyper-parameter n
    */
    public long getNAdjust() {
        return cGetNAdjust(cObject);
    }

    /**
     *  Sets the value of hyper-parameter n
     *  @param nAdjust   Value of hyper-parameter n
     */
    public void setNAdjust(long nAdjust) {
        cSetNAdjust(cObject, nAdjust);
    }

    private native long cInit();
    private native long cGetDimension(long cParameter);
    private native void cSetDimension(long cParameter, long dimension);
    private native double cGetkappa(long cParameter);
    private native void cSetKappa(long cParameter, double kappa);
    private native double cGetAlpha(long cParameter);
    private native void cSetAlpha(long cParameter, double alpha);
    private native double cGetBeta(long cParameter);
    private native void cSetBeta(long cParameter, double beta);
    private native long cGetNAdjust(long cParameter);
    private native void cSetNAdjust(long cParameter, long nAdjust);
}
