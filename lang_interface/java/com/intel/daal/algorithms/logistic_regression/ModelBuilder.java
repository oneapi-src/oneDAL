/* file: ModelBuilder.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @defgroup logistic_regression Logistic Regression
 * @brief Contains classes of the logistic regression algorithm
 * @ingroup regression
 * @{
 */
package com.intel.daal.algorithms.logistic_regression;

import com.intel.daal.utils.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__MODEL__BUILDER"></a>
 * @brief %Class for building model of the logistic regression algorithm
 *
 * @par References
 *      - Parameter class
 */
public class ModelBuilder extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private Precision prec;

    /**
     * Constructs the logistic regression model builder
     * @param context   Context to manage logistic regression model builder
     * @param cls       Data type to use in intermediate computations of logistic regression,
     *                  Double.class or Float.class
     * @param nFeatures Number of features
     * @param nClasses  Number of classes
     */
    public ModelBuilder(DaalContext context, Class<? extends Number> cls, long nFeatures, long nClasses) {
        super(context);
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), nFeatures, nClasses);
    }

    /**
     * Returns built model of logistic regression
     * @return Model of logistic regression
     */
    public Model getModel() {
        return new Model(getContext(), cGetModel(this.cObject, prec.getValue()));
    }

    /**
     * Sets beta-coefficionts to the model
     * @param bufBeta Set of beta-coefficients
     * @param length Size of set of beta-coefficients
     */
    public void setBeta(FloatBuffer bufBeta, long length) {
        final long bufferSize = length;
        float[] betaArr = new float[(int)bufferSize];
        bufBeta.position(0);
        bufBeta.get(betaArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(betaArr);

        cSetBetaFloat(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Sets beta-coefficionts to the model
     * @param bufBeta Set of beta-coefficients
     * @param length Size of set of beta-coefficient
     */
    public void setBeta(DoubleBuffer bufBeta, long length) {
        final long bufferSize = length;
        double[] betaArr = new double[(int)bufferSize];
        bufBeta.position(0);
        bufBeta.get(betaArr);

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(betaArr);

        cSetBetaDouble(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Sets beta-coefficionts to the model
     * @param bufBeta Set of beta-coefficients
     */
    public void setBeta(float [] bufBeta) {
        final long bufferSize = bufBeta.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(bufBeta);

        cSetBetaFloat(this.cObject, byteBuf, bufferSize);
    }

    /**
     * Sets beta-coefficionts to the model
     * @param bufBeta Set of beta-coefficients
     */
    public void setBeta(double [] bufBeta) {
        final long bufferSize = bufBeta.length;

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(bufBeta);

        cSetBetaDouble(this.cObject, byteBuf, bufferSize);
    }

    private native long cInit(int prec, long nFeatures, long nClasses);
    private native long cGetModel(long algAddr, int prec);
    private native void cSetBetaFloat(long algAddr, ByteBuffer byteBuffer, long nBetas);
    private native void cSetBetaDouble(long algAddr, ByteBuffer byteBuffer, long nBetas);
}
/** @} */
