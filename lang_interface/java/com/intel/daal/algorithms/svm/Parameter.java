/* file: Parameter.java */
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
 * @brief Contains classes of the support vector machine (SVM) classification algorithm
 */
package com.intel.daal.algorithms.svm;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__PARAMETER"></a>
 * @brief Optional SVM algorithm parameters
 *
 * @ref opt_notice
 *
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** @private */
    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the SVM algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
        this.cObject = cInitDefault();
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel, double c) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        this.setC(c);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     * @param accuracyThreshold    Accuracy of the SVM training algorithm
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel,
            double c, double accuracyThreshold) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        setC(c);
        setAccuracyThreshold(accuracyThreshold);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     * @param accuracyThreshold    Accuracy of the SVM training algorithm
     * @param tau                  Parameter of the working set selection scheme
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel,
            double c, double accuracyThreshold, double tau) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        setC(c);
        setAccuracyThreshold(accuracyThreshold);
        setTau(tau);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     * @param accuracyThreshold    Accuracy of the SVM training algorithm
     * @param tau                  Parameter of the working set selection scheme
     * @param maxIterations        Maximal number of iterations of the SVM training algorithm
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel,
            double c, double accuracyThreshold, double tau, long maxIterations) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        setC(c);
        setAccuracyThreshold(accuracyThreshold);
        setTau(tau);
        setMaxIterations(maxIterations);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     * @param accuracyThreshold    Accuracy of the SVM training algorithm
     * @param tau                  Parameter of the working set selection scheme
     * @param maxIterations        Maximal number of iterations of the SVM training algorithm
     * @param cacheSize            Size of the cache in bytes
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel,
            double c, double accuracyThreshold, double tau, long maxIterations, long cacheSize) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        setC(c);
        setAccuracyThreshold(accuracyThreshold);
        setTau(tau);
        setMaxIterations(maxIterations);
        setCacheSize(cacheSize);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     * @param accuracyThreshold    Accuracy of the SVM training algorithm
     * @param tau                  Parameter of the working set selection scheme
     * @param maxIterations        Maximal number of iterations of the SVM training algorithm
     * @param cacheSize            Size of the cache in bytes
     * @param doShrinking          Flag that enables use of the shrinking optimization technique
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel,
            double c, double accuracyThreshold, double tau, long maxIterations, long cacheSize,
            boolean doShrinking) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        setC(c);
        setAccuracyThreshold(accuracyThreshold);
        setTau(tau);
        setMaxIterations(maxIterations);
        setCacheSize(cacheSize);
        setDoShrinking(doShrinking);
    }

    /**
     * Constructs a parameter
     * @param context              Context to manage the parameter of the SVM algorithm
     * @param kernel               Kernel function
     * @param c                    Upper bound in constraints of the quadratic optimization problem
     * @param accuracyThreshold    Accuracy of the SVM training algorithm
     * @param tau                  Parameter of the working set selection scheme
     * @param maxIterations        Maximal number of iterations of the SVM training algorithm
     * @param cacheSize            Size of the cache in bytes
     * @param doShrinking          Flag that enables use of the shrinking optimization technique
     * @param shrinkingStep        Number of iterations between the steps of shrinking optimization technique
     */
    public Parameter(DaalContext context, com.intel.daal.algorithms.kernel_function.Batch kernel,
            double c, double accuracyThreshold, double tau, long maxIterations, long cacheSize,
            boolean doShrinking, long shrinkingStep) {
        super(context);
        this.cObject = cInitWithKernel(kernel.cObject);
        setC(c);
        setAccuracyThreshold(accuracyThreshold);
        setTau(tau);
        setMaxIterations(maxIterations);
        setCacheSize(cacheSize);
        setDoShrinking(doShrinking);
        setShrinkingStep(shrinkingStep);
    }

    /**
     * Sets an upper bound in constraints of the quadratic optimization problem
     * @param C     Upper bound in constraints of the quadratic optimization problem
     */
    public void setC(double C) {
        cSetC(this.cObject, C);
    }

    /**
     * Retrieves an upper bound in constraints of the quadratic optimization problem
     * @return Upper bound in constraints of the quadratic optimization problem
     */
    public double getC() {
        return cGetC(this.cObject);
    }

    /**
     * Sets the accuracy of the SVM training algorithm
     * @param accuracyThreshold     Accuracy of the SVM training algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Retrieves the accuracy of the SVM training algorithm
     * @return Accuracy of the SVM training algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the tau parameter of the working set selection scheme
     * @param tau     Parameter of the working set selection scheme
     */
    public void setTau(double tau) {
        cSetTau(this.cObject, tau);
    }

    /**
     * Retrieves the tau parameter of the working set selection scheme
     * @return Parameter of the working set selection scheme
     */
    public double getTau() {
        return cGetTau(this.cObject);
    }

    /**
     * Sets the maximal number of iterations of the SVM training algorithm
     * @param maxIterations Maximal number of iterations of the SVM training algorithm
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Retrieves the maximal number of iterations of the SVM training algorithm
     * @return Maximal number of iterations of the SVM training algorithm
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Sets the size of the cache in bytes to store values of the kernel matrix.
     * A non-zero value enables use of a cache optimization technique
     * @param cacheSize Size of the cache in bytes
     */
    public void setCacheSize(long cacheSize) {
        cSetCacheSize(this.cObject, cacheSize);
    }

    /**
     * Retrieves the size of the cache in bytes to store values of the kernel matrix.
     * @return Size of the cache in bytes
     */
    public long getCacheSize() {
        return cGetCacheSize(this.cObject);
    }

    /**
     * Sets the flag that enables use of the shrinking optimization technique
     * @param doShrinking   Flag that enables use of the shrinking optimization technique
     */
    public void setDoShrinking(boolean doShrinking) {
        cSetDoShrinking(this.cObject, doShrinking);
    }

    /**
     * Retrieves the flag that enables use of the shrinking optimization technique
     * @return   Flag that enables use of the shrinking optimization technique
     */
    public boolean getDoShrinking() {
        return cGetDoShrinking(this.cObject);
    }

    /**
     * Sets the number of iterations between the steps of shrinking optimization technique
     * @param shrinkingStep   Number of iterations between the steps of shrinking optimization technique
     */
    public void setShrinkingStep(long shrinkingStep) {
        cSetShrinkingStep(this.cObject, shrinkingStep);
    }

    /**
     * Retrieves the number of iterations between the steps of shrinking optimization technique
     * @return   Number of iterations between the steps of shrinking optimization technique
     */
    public long getShrinkingStep() {
        return cGetShrinkingStep(this.cObject);
    }

    /**
     * Sets the kernel function
     * @param kernel    Kernel function
     */
    public void setKernel(com.intel.daal.algorithms.kernel_function.Batch kernel) {
        cSetKernel(this.cObject, kernel.cObject);
    }


    private native long cInitDefault();

    private native long cInitWithKernel(long kernelAddr);

    private native void cSetC(long parAddr, double C);

    private native double cGetC(long parAddr);

    private native void cSetAccuracyThreshold(long parAddr, double Eps);

    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetTau(long parAddr, double Tau);

    private native double cGetTau(long parAddr);

    private native void cSetMaxIterations(long parAddr, long Iter);

    private native long cGetMaxIterations(long parAddr);

    private native void cSetCacheSize(long parAddr, long CacheSize);

    private native long cGetCacheSize(long parAddr);

    private native void cSetDoShrinking(long parAddr, boolean DoShrinking);

    private native boolean cGetDoShrinking(long parAddr);

    private native void cSetShrinkingStep(long parAddr, long ShrinkingStep);

    private native long cGetShrinkingStep(long parAddr);

    private native void cSetKernel(long parAddr, long kernelAddr);
}
