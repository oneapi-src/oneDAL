/* file: LrnParameter.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup lrn
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lrn;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LRNPARAMETER"></a>
 * \brief Class that specifies parameters of the local response normalization layer
 */
public class LrnParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     * Constructs the parameter of the local response normalization layer
     * @param context   Context to manage the parameter of the local response normalization layer
     */
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
/** @} */
