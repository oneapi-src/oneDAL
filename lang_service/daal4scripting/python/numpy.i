/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/* -*- C -*-  (not really, but good for syntax highlighting) */

// Derived from numpy.i

/*
 * Copyright (c) 2005-2015, NumPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials provided
 *        with the distribution.
 *
 *     * Neither the name of the NumPy Developers nor the names of any
 *        contributors may be used to endorse or promote products derived
 *        from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef SWIGPYTHON

%{
#ifndef SWIG_FILE_WITH_INIT
#define NO_IMPORT_ARRAY
#endif
#include "stdio.h"
#include <cassert>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define FORCE_INPLACE true
#define ALLOW_CONVERSION false
%}

/**********************************************************************/

%fragment("NumPy_Backward_Compatibility", "header")
{
%#if NPY_API_VERSION < 0x00000007
%#define NPY_ARRAY_DEFAULT NPY_DEFAULT
%#define NPY_ARRAY_FARRAY  NPY_FARRAY
%#define NPY_FORTRANORDER  NPY_FORTRAN
%#endif
}

/**********************************************************************/

/* The following code originally appeared in
 * enthought/kiva/agg/src/numeric.i written by Eric Jones.  It was
 * translated from C++ to C by John Hunter.  Bill Spotz has modified
 * it to fix some minor bugs, upgrade from Numeric to numpy (all
 * versions), add some comments and functionality, and convert from
 * direct code insertion to SWIG fragments.
 */

%fragment("NumPy_Macros", "header")
{
/* Macros to extract array attributes.
 */
%#if NPY_API_VERSION < 0x00000007
%#define is_array(a)            ((a) && PyArray_Check((PyArrayObject*)a))
%#define array_type(a)          (int)(PyArray_TYPE((PyArrayObject*)a))
%#define array_numdims(a)       (((PyArrayObject*)a)->nd)
%#define array_dimensions(a)    (((PyArrayObject*)a)->dimensions)
%#define array_size(a,i)        (((PyArrayObject*)a)->dimensions[i])
%#define array_strides(a)       (((PyArrayObject*)a)->strides)
%#define array_stride(a,i)      (((PyArrayObject*)a)->strides[i])
%#define array_data(a)          (((PyArrayObject*)a)->data)
%#define array_descr(a)         (((PyArrayObject*)a)->descr)
%#define array_flags(a)         (((PyArrayObject*)a)->flags)
%#define array_enableflags(a,f) (((PyArrayObject*)a)->flags) = f
%#else
%#define is_array(a)            ((a) && PyArray_Check(a))
%#define array_type(a)          PyArray_TYPE((PyArrayObject*)a)
%#define array_numdims(a)       PyArray_NDIM((PyArrayObject*)a)
%#define array_dimensions(a)    PyArray_DIMS((PyArrayObject*)a)
%#define array_strides(a)       PyArray_STRIDES((PyArrayObject*)a)
%#define array_stride(a,i)      PyArray_STRIDE((PyArrayObject*)a,i)
%#define array_size(a,i)        PyArray_DIM((PyArrayObject*)a,i)
%#define array_data(a)          PyArray_DATA((PyArrayObject*)a)
%#define array_descr(a)         PyArray_DESCR((PyArrayObject*)a)
%#define array_flags(a)         PyArray_FLAGS((PyArrayObject*)a)
%#define array_enableflags(a,f) PyArray_ENABLEFLAGS((PyArrayObject*)a,f)
%#endif
%#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS((PyArrayObject*)a))
%#define array_is_native(a)     (PyArray_ISNOTSWAPPED((PyArrayObject*)a))
%#define array_is_fortran(a)    (PyArray_ISFORTRAN((PyArrayObject*)a))
}

/**********************************************************************/

%fragment("NumPy_Utilities",
          "header")
{
  /* Given a PyObject, return a string describing its type.
   */
  const char* pytype_string(PyObject* py_obj)
  {
    if (py_obj == NULL          ) return "C NULL value";
    if (py_obj == Py_None       ) return "Python None" ;
    if (PyCallable_Check(py_obj)) return "callable"    ;
    if (PyString_Check(  py_obj)) return "string"      ;
    if (PyInt_Check(     py_obj)) return "int"         ;
    if (PyFloat_Check(   py_obj)) return "float"       ;
    if (PyDict_Check(    py_obj)) return "dict"        ;
    if (PyList_Check(    py_obj)) return "list"        ;
    if (PyTuple_Check(   py_obj)) return "tuple"       ;
%#if PY_MAJOR_VERSION < 3
    if (PyFile_Check(    py_obj)) return "file"        ;
    if (PyModule_Check(  py_obj)) return "module"      ;
    if (PyInstance_Check(py_obj)) return "instance"    ;
%#endif

    return "unkown type";
  }

  /* Given a NumPy typecode, return a string describing the type.
   */
  const char* typecode_string(int typecode)
  {
    static const char* type_names[25] = {"bool",
                                         "byte",
                                         "unsigned byte",
                                         "short",
                                         "unsigned short",
                                         "int",
                                         "unsigned int",
                                         "long",
                                         "unsigned long",
                                         "long long",
                                         "unsigned long long",
                                         "float",
                                         "double",
                                         "long double",
                                         "complex float",
                                         "complex double",
                                         "complex long double",
                                         "object",
                                         "string",
                                         "unicode",
                                         "void",
                                         "ntypes",
                                         "notype",
                                         "char",
                                         "unknown"};
    return typecode < 24 ? type_names[typecode] : type_names[24];
  }

  /* Make sure input has correct numpy type.  This now just calls
     PyArray_EquivTypenums().
   */
  int type_match(int actual_type,
                 int desired_type)
  {
    return PyArray_EquivTypenums(actual_type, desired_type);
  }

%#ifdef SWIGPY_USE_CAPSULE
  void free_cap(PyObject * cap)
  {
    void* array = (void*) PyCapsule_GetPointer(cap,SWIGPY_CAPSULE_NAME);
    if (array != NULL) free(array);
  }
%#endif


}

/**********************************************************************/

%fragment("NumPy_Object_to_Array",
          "header",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros",
          fragment="NumPy_Utilities")
{
  /* Given a PyObject pointer, cast it to a PyArrayObject pointer if
   * legal.  If not, set the python error string appropriately and
   * return NULL.
   */
  PyArrayObject* obj_to_array_no_conversion(PyObject* input,
                                            int        typecode)
  {
    PyArrayObject* ary = NULL;
    if (is_array(input) && (typecode == NPY_NOTYPE ||
                            PyArray_EquivTypenums(array_type(input), typecode)))
    {
      ary = (PyArrayObject*) input;
    }
    else if is_array(input)
    {
      const char* desired_type = typecode_string(typecode);
      const char* actual_type  = typecode_string(array_type(input));
      PyErr_Format(PyExc_TypeError,
                   "Array of type '%s' required.  Array of type '%s' given",
                   desired_type, actual_type);
      ary = NULL;
    }
    else
    {
      const char* desired_type = typecode_string(typecode);
      const char* actual_type  = pytype_string(input);
      PyErr_Format(PyExc_TypeError,
                   "Array of type '%s' required.  A '%s' was given",
                   desired_type,
                   actual_type);
      ary = NULL;
    }
    return ary;
  }

  /* Convert the given PyObject to a NumPy array with the given
   * typecode.  On success, return a valid PyArrayObject* with the
   * correct type.  On failure, the python error string will be set and
   * the routine returns NULL.
   */
  PyArrayObject* obj_to_array_allow_conversion(PyObject* input,
                                               int       typecode,
                                               int*      is_new_object)
  {
    PyArrayObject* ary = NULL;
    PyObject*      py_obj;
    if (is_array(input) && (typecode == NPY_NOTYPE ||
                            PyArray_EquivTypenums(array_type(input),typecode)))
    {
      ary = (PyArrayObject*) input;
      *is_new_object = 0;
    }
    else
    {
      py_obj = PyArray_FROMANY(input, typecode, 0, 0, NPY_ARRAY_DEFAULT);
      /* If NULL, PyArray_FromObject will have set python error value.*/
      ary = (PyArrayObject*) py_obj;
      *is_new_object = 1;
    }
    return ary;
  }

  /* Given a PyArrayObject, check to see if it is contiguous.  If so,
   * return the input pointer and flag it as not a new object.  If it is
   * not contiguous, create a new PyArrayObject using the original data,
   * flag it as a new object and return the pointer.
   */
  PyArrayObject* make_contiguous(PyArrayObject* ary,
                                 int*           is_new_object,
                                 int            min_dims,
                                 int            max_dims)
  {
    PyArrayObject* result;
    if (array_is_contiguous(ary))
    {
      result = ary;
      *is_new_object = 0;
    }
    else
    {
      result = (PyArrayObject*) PyArray_ContiguousFromObject((PyObject*)ary,
                                                              array_type(ary),
                                                              min_dims,
                                                              max_dims);
      *is_new_object = 1;
    }
    return result;
  }

  /* Given a PyArrayObject, check to see if it is Fortran-contiguous.
   * If so, return the input pointer, but do not flag it as not a new
   * object.  If it is not Fortran-contiguous, create a new
   * PyArrayObject using the original data, flag it as a new object
   * and return the pointer.
   */
  PyArrayObject* make_fortran(PyArrayObject* ary,
                              int*           is_new_object)
  {
    PyArrayObject* result;
    if (array_is_fortran(ary))
    {
      result = ary;
      *is_new_object = 0;
    }
    else
    {
      Py_INCREF(array_descr(ary));
      result = (PyArrayObject*) PyArray_FromArray(ary,
                                                  array_descr(ary),
                                                  NPY_FORTRANORDER);
      *is_new_object = 1;
    }
    return result;
  }

  /* Convert a given PyObject to a contiguous PyArrayObject of the
   * specified type.  If the input object is not a contiguous
   * PyArrayObject, a new one will be created and the new object flag
   * will be set.
   */
  PyArrayObject* obj_to_array_contiguous_allow_conversion(PyObject* input,
                                                          int       typecode,
                                                          int*      is_new_object)
  {
    int is_new1 = 0;
    int is_new2 = 0;
    PyArrayObject* ary2;
    PyArrayObject* ary1 = obj_to_array_allow_conversion(input,
                                                        typecode,
                                                        &is_new1);
    if (ary1)
    {
      ary2 = make_contiguous(ary1, &is_new2, 0, 0);
      if ( is_new1 && is_new2)
      {
        Py_DECREF(ary1);
      }
      ary1 = ary2;
    }
    *is_new_object = is_new1 || is_new2;
    return ary1;
  }

  /* Convert a given PyObject to a Fortran-ordered PyArrayObject of the
   * specified type.  If the input object is not a Fortran-ordered
   * PyArrayObject, a new one will be created and the new object flag
   * will be set.
   */
  PyArrayObject* obj_to_array_fortran_allow_conversion(PyObject* input,
                                                       int       typecode,
                                                       int*      is_new_object)
  {
    int is_new1 = 0;
    int is_new2 = 0;
    PyArrayObject* ary2;
    PyArrayObject* ary1 = obj_to_array_allow_conversion(input,
                                                        typecode,
                                                        &is_new1);
    if (ary1)
    {
      ary2 = make_fortran(ary1, &is_new2);
      if (is_new1 && is_new2)
      {
        Py_DECREF(ary1);
      }
      ary1 = ary2;
    }
    *is_new_object = is_new1 || is_new2;
    return ary1;
  }

/* ****************************************************************************
   * Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) extensions  *
   ****************************************************************************/
  class NumpyDeleter : public daal::services::DeleterIface
  {
  public:
      // constructor to initialize with ndarray
      NumpyDeleter(PyArrayObject* a) : _ndarray(a) {}
      // DeleterIface must be copy-constrible
      NumpyDeleter(const NumpyDeleter & o) : _ndarray(o._ndarray) {}
      // ref-count reached 0 -> decref reference to python object
      void operator() (const void *ptr) DAAL_C11_OVERRIDE
      {
          // This gets called from destructors, the SWIG wrappers release the GIL
          // -> need to protect calls to python API
          // Note: at termination time, even when no threads are running, this breaks without the protection
          SWIG_PYTHON_THREAD_BEGIN_BLOCK;
          assert((void *)array_data(_ndarray) == ptr);
          Py_DECREF(_ndarray);
          SWIG_PYTHON_THREAD_END_BLOCK;
      }
  private:
      PyArrayObject* _ndarray;
  };

  // An empty virtual base class (used by TVSP) for shared pointer handling
  // we use this to have a generic type for all shared pointers
  // e.g. used in daalsp_free functions below
  class VSP 
  {
  public:
      // we need a virtual destructor
      virtual ~VSP() {};
  };
  // typed virtual shared pointer, for simplicity we make it a DAAL shared pointer
  template< typename T >
  class TVSP : public VSP, public daal::services::SharedPtr<T>
  {
  public:
      TVSP(const daal::services::SharedPtr<T> & org) : daal::services::SharedPtr<T>(org) {}
      virtual ~TVSP() {};
  };

  // define our own free functions for wrapping python objects holding our shared pointers
%#ifdef SWIGPY_USE_CAPSULE
  void daalsp_free_cap(PyObject * cap)
  {
      VSP * sp = (VSP*) PyCapsule_GetPointer(cap,SWIGPY_CAPSULE_NAME);
      if (sp) delete sp;
  }
%#else
  void daalsp_free(PyObject * cap)
  {
      VSP * sp = (VSP*) PyCObject_AsVoidPtr(cap);
      if (sp) delete sp;
  }
%#endif
     
  template< typename T >
  void set_sp_base(PyArrayObject * ary, daal::services::SharedPtr<T> & sp)
  {
      void * tmp_sp = (void*) new TVSP<T>(sp);
%#ifdef SWIGPY_USE_CAPSULE
      PyObject* cap = PyCapsule_New(tmp_sp, SWIGPY_CAPSULE_NAME, daalsp_free_cap);
%#else
      PyObject* cap = PyCObject_FromVoidPtr(tmp_sp, daalsp_free);
%#endif
      PyArray_SetBaseObject(ary, cap);
    }

/* ****************************************************************************
   * Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) extensions  *
   ****************************************************************************/

} /* end fragment */

/**********************************************************************/

%fragment("NumPy_Array_Requirements",
          "header",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros")
{
  /* Test whether a python object is contiguous.  If array is
   * contiguous, return 1.  Otherwise, set the python error string and
   * return 0.
   */
  int require_contiguous(PyArrayObject* ary)
  {
    int contiguous = 1;
    if (!array_is_contiguous(ary))
    {
      PyErr_SetString(PyExc_TypeError,
                      "Array must be contiguous.  A non-contiguous array was given");
      contiguous = 0;
    }
    return contiguous;
  }

  /* Test whether a python object is (C_ or F_) contiguous.  If array is
   * contiguous, return 1.  Otherwise, set the python error string and
   * return 0.
   */
  int require_c_or_f_contiguous(PyArrayObject* ary)
  {
    int contiguous = 1;
    if (!(array_is_contiguous(ary) || array_is_fortran(ary)))
    {
      PyErr_SetString(PyExc_TypeError,
                      "Array must be contiguous (C_ or F_).  A non-contiguous array was given");
      contiguous = 0;
    }
    return contiguous;
  }

  /* Require that a numpy array is not byte-swapped.  If the array is
   * not byte-swapped, return 1.  Otherwise, set the python error string
   * and return 0.
   */
  int require_native(PyArrayObject* ary)
  {
    int native = 1;
    if (!array_is_native(ary))
    {
      PyErr_SetString(PyExc_TypeError,
                      "Array must have native byteorder.  "
                      "A byte-swapped array was given");
      native = 0;
    }
    return native;
  }

  /* Require the given PyArrayObject to have a specified number of
   * dimensions.  If the array has the specified number of dimensions,
   * return 1.  Otherwise, set the python error string and return 0.
   */
  int require_dimensions(PyArrayObject* ary,
                         int            exact_dimensions)
  {
    int success = 1;
    if (array_numdims(ary) != exact_dimensions)
    {
      PyErr_Format(PyExc_TypeError,
                   "Array must have %d dimensions.  Given array has %d dimensions",
                   exact_dimensions,
                   array_numdims(ary));
      success = 0;
    }
    return success;
  }

  /* Require the given PyArrayObject to have one of a list of specified
   * number of dimensions.  If the array has one of the specified number
   * of dimensions, return 1.  Otherwise, set the python error string
   * and return 0.
   */
  int require_dimensions_n(PyArrayObject* ary,
                           int*           exact_dimensions,
                           int            n)
  {
    int success = 0;
    int i;
    char dims_str[255] = "";
    char s[255];
    for (i = 0; i < n && !success; i++)
    {
      if (array_numdims(ary) == exact_dimensions[i])
      {
        success = 1;
      }
    }
    if (!success)
    {
      for (i = 0; i < n-1; i++)
      {
        sprintf(s, "%d, ", exact_dimensions[i]);
        strcat(dims_str,s);
      }
      sprintf(s, " or %d", exact_dimensions[n-1]);
      strcat(dims_str,s);
      PyErr_Format(PyExc_TypeError,
                   "Array must have %s dimensions.  Given array has %d dimensions",
                   dims_str,
                   array_numdims(ary));
    }
    return success;
  }

  /* Require the given PyArrayObject to have a specified shape.  If the
   * array has the specified shape, return 1.  Otherwise, set the python
   * error string and return 0.
   */
  int require_size(PyArrayObject* ary,
                   npy_intp*      size,
                   int            n)
  {
    int i;
    int success = 1;
    int len;
    char desired_dims[255] = "[";
    char s[255];
    char actual_dims[255] = "[";
    for(i=0; i < n;i++)
    {
      if (size[i] != -1 &&  size[i] != array_size(ary,i))
      {
        success = 0;
      }
    }
    if (!success)
    {
      for (i = 0; i < n; i++)
      {
        if (size[i] == -1)
        {
          sprintf(s, "*,");
        }
        else
        {
          sprintf(s, "%ld,", (long int)size[i]);
        }
        strcat(desired_dims,s);
      }
      len = strlen(desired_dims);
      desired_dims[len-1] = ']';
      for (i = 0; i < n; i++)
      {
        sprintf(s, "%ld,", (long int)array_size(ary,i));
        strcat(actual_dims,s);
      }
      len = strlen(actual_dims);
      actual_dims[len-1] = ']';
      PyErr_Format(PyExc_TypeError,
                   "Array must have shape of %s.  Given array has shape of %s",
                   desired_dims,
                   actual_dims);
    }
    return success;
  }

  /* Require the given PyArrayObject to to be Fortran ordered.  If the
   * the PyArrayObject is already Fortran ordered, do nothing.  Else,
   * set the Fortran ordering flag and recompute the strides.
   */
  int require_fortran(PyArrayObject* ary)
  {
    int success = 1;
    int nd = array_numdims(ary);
    int i;
    npy_intp * strides = array_strides(ary);
    if (array_is_fortran(ary)) return success;
    /* Set the Fortran ordered flag */
    array_enableflags(ary,NPY_ARRAY_FARRAY);
    /* Recompute the strides */
    strides[0] = strides[nd-1];
    for (i=1; i < nd; ++i)
      strides[i] = strides[i-1] * array_size(ary,i-1);
    return success;
  }
}

/* Combine all NumPy fragments into one for convenience */
%fragment("NumPy_Fragments",
          "header",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros",
          fragment="NumPy_Utilities",
          fragment="NumPy_Object_to_Array",
          fragment="NumPy_Array_Requirements")
{
}

/* End John Hunter translation (with modifications by Bill Spotz)
 */

/*
  We support the following typemaps (to be used with %apply).
  The input typemaps have 2 variants:
    1. ${INPLACE}==FORCE_INPLACE will not allow any conversion of the input array
    2. ${INPLACE}==ALLOW_CONVERSION will try to use the input array in-place.
       If that's not possible it will try creating a new array and copy/convert the data.

  (DATA_TYPE* IN_${INPLACE}_ARRAY1, DIM_TYPE DIM)
  (DIM_TYPE DIM, DATA_TYPE* IN_${INPLACE}_ARRAY1)
  (const daal::services::SharedPtr< DATA_TYPE > & IN_${INPLACE}_SP_ARRAY1, DIM_TYPE DIM)
  (const daal::services::SharedPtr< DATA_TYPE > & IN_${INPLACE}_SP_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2)
  (const daal::services::SharedPtr< DATA_TYPE > & IN_${INPLACE}_SP_SYM_ARRAY2, DIM_TYPE DIM)
  (size_t N, const DIM_TYPE * DIMS, const daal::services::SharedPtr< DATA_TYPE > & IN_${INPLACE}_SP_N_DIMS_ARRAY)

  (daal::data_management::BlockDescriptor<DATA_TYPE> ** ARGOUTVIEW_DBD_ARRAY2)
  (daal::services::Collection<size_t> const** DIMS, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_DIMS_ARRAY)
  (daal::services::SharedPtr< DATA_TYPE > * ARGOUTVIEW_SP_ARRAY1, DIM_TYPE* DIM)
  (DIM_TYPE* DIM, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY1)
  (daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2)
  (DIM_TYPE *NDIMS, DIM_TYPE **DIMSIZES, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_N_DIMS_ARRAY)
*/

// ************************************
// Input typemaps
// ************************************
%define %numpy_in_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE, INPLACE)

/* Typemap suite for (DIM_TYPE DIM, DATA_TYPE* IN_ ## INPLACE ## _ARRAY1)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
  (DIM_TYPE DIM, DATA_TYPE* IN_ ## INPLACE ## _ARRAY1)
{
    $1 = (INPLACE == FORCE_INPLACE
          ? (is_array($input) && PyArray_EquivTypenums(array_type($input), DATA_TYPECODE))
          : (is_array($input) || PySequence_Check($input)));
}
%typemap(in, fragment="NumPy_Fragments")
  (DIM_TYPE DIM, DATA_TYPE* IN_ ## INPLACE ## _ARRAY1)
  (PyArrayObject* array=NULL, int is_new_object=0)
{
  array = (INPLACE == FORCE_INPLACE
           ? obj_to_array_no_conversion($input, DATA_TYPECODE)
           : obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE, &is_new_object));
  if (!array || !require_dimensions(array, 1) || !require_contiguous(array) || !require_native(array)) SWIG_fail;
  $1 = (DIM_TYPE) array_size(array,0);
  $2 = (DATA_TYPE*) array_data(array);
}
%typemap(freearg)
  (DIM_TYPE DIM, DATA_TYPE* IN_ ## INPLACE ## _ARRAY1)
{
    if(INPLACE != FORCE_INPLACE && is_new_object$argnum && array$argnum) {
        Py_DECREF(array$argnum);
    }
}

/* Typemap suite for (DATA_TYPE* IN_ ## INPLACE ## _ARRAY1, DIM_TYPE DIM)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
  (DATA_TYPE* IN_ ## INPLACE ## _ARRAY1, DIM_TYPE DIM)
{
    $1 = (INPLACE == FORCE_INPLACE
          ? (is_array($input) && PyArray_EquivTypenums(array_type($input), DATA_TYPECODE))
          : (is_array($input) || PySequence_Check($input)));
}
%typemap(in, fragment="NumPy_Fragments")
  (DATA_TYPE* IN_ ## INPLACE ## _ARRAY1, DIM_TYPE DIM)
  (PyArrayObject* array=NULL, int i=1, int is_new_object=0)
{
    array = (INPLACE == FORCE_INPLACE
             ? obj_to_array_no_conversion($input, DATA_TYPECODE)
             : obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE, &is_new_object));
    if (!array || !require_dimensions(array, 1) || !require_contiguous(array) || !require_native(array)) SWIG_fail;
    $1 = (DATA_TYPE*) array_data(array);
    $2 = (DIM_TYPE) array_size(array,0);
}
%typemap(freearg)
  (DATA_TYPE* IN_ ## INPLACE ## _ARRAY1, DIM_TYPE DIM)
{
    if(INPLACE != FORCE_INPLACE && is_new_object$argnum && array$argnum) {
        Py_DECREF(array$argnum);
    }
}

/* Typemap suite for (daal::services::SharedPtr< DATA_TYPE >* IN_ ## INPLACE ## _SP_ARRAY1, DIM_TYPE DIM)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
  (const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_ARRAY1, DIM_TYPE DIM)
{
    $1 = (INPLACE == FORCE_INPLACE
          ? (is_array($input) && PyArray_EquivTypenums(array_type($input), DATA_TYPECODE))
          : (is_array($input) || PySequence_Check($input)));
}
%typemap(in, fragment="NumPy_Fragments")
  (const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_ARRAY1, DIM_TYPE DIM)
  (daal::services::SharedPtr< DATA_TYPE > r4p)
{
    int is_new_object = 0;
    PyArrayObject* array = (INPLACE == FORCE_INPLACE
                            ? obj_to_array_no_conversion($input, DATA_TYPECODE)
                            : obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE, &is_new_object));
    if (!array || !require_dimensions(array, 1) || !require_contiguous(array) || !require_native(array)) SWIG_fail;
    // we provide the SharedPtr with a deleter which decrements the pyref
    r4p = daal::services::SharedPtr< DATA_TYPE >((DATA_TYPE*)array_data(array), NumpyDeleter(array));
    $1 = &r4p;
    $2 = (DIM_TYPE) array_size(array,0);
    // we need it increment the ref-count if we use the $input array in-place
    // if we copied/converted it we already own our own reference
    if((PyObject*)array == $input) Py_INCREF(array);
}

/* Typemap suite for (DATA_TYPE* IN_ ## INPLACE ## _SP_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
    (const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2)
{
    $1 = (INPLACE == FORCE_INPLACE
          ? (is_array($input) && PyArray_EquivTypenums(array_type($input), DATA_TYPECODE))
          : (is_array($input) || PySequence_Check($input)));
}
%typemap(in, fragment="NumPy_Fragments")
  (const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2)
  (daal::services::SharedPtr< DATA_TYPE > r4p)
{
    int is_new_object = 0;
    PyArrayObject* array = (INPLACE == FORCE_INPLACE
                            ? obj_to_array_no_conversion($input, DATA_TYPECODE)
                            : obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE, &is_new_object));
    if (!array || !require_dimensions(array, 2) || !require_contiguous(array) || !require_native(array)) SWIG_fail;
    // we provide the SharedPtr with a deleter which decrements the pyref
    r4p = daal::services::SharedPtr< DATA_TYPE >((DATA_TYPE*)array_data(array), NumpyDeleter(array));
    $1 = &r4p;
    $2 = (DIM_TYPE) array_size(array,1);
    $3 = (DIM_TYPE) array_size(array,0);
    // we need it increment the ref-count if we use the $input array in-place
    // if we copied/converted it we already own our own reference
    if((PyObject*)array == $input) Py_INCREF(array);
}

/* Typemap suite for (DATA_TYPE* IN_ ## INPLACE ## _SP_SYM_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
    (const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_SYM_ARRAY2, DIM_TYPE DIM)
{
    $1 = (INPLACE == FORCE_INPLACE
          ? (is_array($input) && PyArray_EquivTypenums(array_type($input), DATA_TYPECODE))
          : (is_array($input) || PySequence_Check($input)));
}
%typemap(in, fragment="NumPy_Fragments")
  (const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_SYM_ARRAY2, DIM_TYPE DIM)
  (daal::services::SharedPtr< DATA_TYPE > r4p)
{
    int is_new_object = 0;
    PyArrayObject* array = (INPLACE == FORCE_INPLACE
                            ? obj_to_array_no_conversion($input, DATA_TYPECODE)
                            : obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE, &is_new_object));
    if (!array || !require_dimensions(array, 2) || !require_contiguous(array) || !require_native(array)) SWIG_fail;
    // we provide the SharedPtr with a deleter which decrements the pyref
    r4p = daal::services::SharedPtr< DATA_TYPE >((DATA_TYPE*)array_data(array), NumpyDeleter(array));
    $1 = &r4p;
    $2 = (DIM_TYPE) array_size(array,0);
    // we need it increment the ref-count if we use the $input array in-place
    // if we copied/converted it we already own our own reference
    if((PyObject*)array == $input) Py_INCREF(array);
}

/* Typemap suite for (size_t N, const DIM_TYPE *DIMS, daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_N_DIMS_ARRAY)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
    (size_t N, const DIM_TYPE * DIMS, const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_N_DIMS_ARRAY)
{
    $1 = (INPLACE == FORCE_INPLACE
          ? (is_array($input) && PyArray_EquivTypenums(array_type($input), DATA_TYPECODE))
          : (is_array($input) || PySequence_Check($input)));
}
%typemap(in, fragment="NumPy_Fragments")
  (size_t N, const DIM_TYPE * DIMS, const daal::services::SharedPtr< DATA_TYPE > & IN_ ## INPLACE ## _SP_N_DIMS_ARRAY)
  (daal::services::SharedPtr< DATA_TYPE > r4p)
{
    int is_new_object = 0;
    PyArrayObject * array = (INPLACE == FORCE_INPLACE
                             ? obj_to_array_no_conversion($input, DATA_TYPECODE)
                             : obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE, &is_new_object));
    if (!array || !require_contiguous(array) || !require_native(array)) SWIG_fail;
    $1 = (DIM_TYPE)array_numdims(array);
    $2 = (DIM_TYPE*)array_dimensions(array);
    // we provide the SharedPtr with a deleter which decrements the pyref
    r4p = daal::services::SharedPtr< DATA_TYPE >((DATA_TYPE*)array_data(array), NumpyDeleter(array));
    $3 = &r4p;
    // we need it increment the ref-count if we use the $input array in-place
    // if we copied/converted it we already own our own reference
    if((PyObject*)array == $input) Py_INCREF(array);
}

%enddef    /* %numpy_in_typemaps() macro */

// ************************************
// Outarg typemaps
// ************************************

%define %numpy_argout_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE)
/* Typemap suite for (DIM_TYPE* DIM, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY1)
 */
%typemap(in,numinputs=0)
    (DIM_TYPE* DIM, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY1)
    (daal::services::SharedPtr< DATA_TYPE > data_temp, DIM_TYPE  dim_temp)
{
  $1 = &dim_temp;
  $2 = &data_temp;
}
%typemap(argout, fragment="NumPy_Backward_Compatibility")
    (DIM_TYPE* DIM, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY1)
{
    npy_intp dims[1] = { static_cast<npy_intp>(*$1) }; // Intel/FS static-cast
    PyObject* obj = PyArray_SimpleNewFromData(1, dims, DATA_TYPECODE, (void*)($2->get()));
    if (!obj) SWIG_fail;
    set_sp_base((PyArrayObject*)obj, *$2);
    $result = SWIG_Python_AppendOutput($result,obj);
}

/* Typemap suite for (DIM_TYPE* DIM, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY1)
 */
%typemap(in,numinputs=0)
     (daal::services::SharedPtr< DATA_TYPE > * ARGOUTVIEW_SP_ARRAY1, DIM_TYPE* DIM)
     (daal::services::SharedPtr< DATA_TYPE > data_temp, DIM_TYPE  dim_temp)
{
  $1 = &data_temp;
  $2 = &dim_temp;
}
%typemap(argout, fragment="NumPy_Backward_Compatibility")
     (daal::services::SharedPtr< DATA_TYPE > * ARGOUTVIEW_SP_ARRAY1, DIM_TYPE* DIM)
{
    npy_intp dims[1] = { static_cast<npy_intp>(*$2) }; // Intel/FS static-cast
    PyObject* obj = PyArray_SimpleNewFromData(1, dims, DATA_TYPECODE, (void*)($1->get()));
    if (!obj) SWIG_fail;
    set_sp_base((PyArrayObject*)obj, *$1);
    $result = SWIG_Python_AppendOutput($result,obj);
}

/* Typemap suite for (DATA_TYPE** ARGOUTVIEW_SP_ARRAY2, DIM_TYPE* DIM, DIM_TYPE* DIM2)
 */
%typemap(in,numinputs=0)
    (daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2     )
    (daal::services::SharedPtr< DATA_TYPE > data_temp, DIM_TYPE  dim1_temp, DIM_TYPE  dim2_temp)
{
    $1 = &data_temp;
    $2 = &dim1_temp;
    $3 = &dim2_temp;
}
%typemap(argout, fragment="NumPy_Backward_Compatibility")
    (daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2)
{
    npy_intp dims[2] = { static_cast<npy_intp>(*$3), static_cast<npy_intp>(*$2) };
    PyObject* obj = PyArray_SimpleNewFromData(2, dims, DATA_TYPECODE, (void*)($1->get()));
    if (!obj) SWIG_fail;
    set_sp_base((PyArrayObject*)obj, *$1);
    $result = SWIG_Python_AppendOutput($result,obj);
}

/* Typemap suite for (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DATA_TYPE** ARGOUTVIEW_DBD_ARRAY2)
   ATTENTION: deletes incoming BlockDescriptor!
 */
%typemap(in,numinputs=0)
    (daal::data_management::BlockDescriptor<DATA_TYPE> ** ARGOUTVIEW_DBD_ARRAY2)
    (daal::data_management::BlockDescriptor<DATA_TYPE> * data_temp)
{
  $1 = &data_temp;
}
%typemap(argout, fragment="NumPy_Backward_Compatibility,NumPy_Utilities")
    (daal::data_management::BlockDescriptor<DATA_TYPE> ** ARGOUTVIEW_DBD_ARRAY2)
{
    npy_intp dims[2] = { static_cast<npy_intp>((*$1)->getNumberOfRows()), static_cast<npy_intp>((*$1)->getNumberOfColumns()) };
    daal::services::SharedPtr< DATA_TYPE > data_tmp((*$1)->getBlockSharedPtr());
    PyObject* obj = PyArray_SimpleNewFromData(2, dims, DATA_TYPECODE, (void*)data_tmp.get());
    if (!obj) SWIG_fail;
    set_sp_base((PyArrayObject*)obj, data_tmp);
    $result = SWIG_Python_AppendOutput($result, obj);
    // oooohhhh
    delete *$1;
}


/* Typemap suite for (DIM_TYPE *NDIMS, DIM_TYPE **DIMSIZES, DATA_TYPE** ARGOUTVIEW_SP_N_DIMS_ARRAY)
 */
%typemap(in,numinputs=0)
    (DIM_TYPE *NDIMS, DIM_TYPE **DIMSIZES, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_N_DIMS_ARRAY)
    (DIM_TYPE ndims_tmp, DIM_TYPE *dimsizes_tmp,daal::services::SharedPtr< DATA_TYPE > data_tmp)
{
    $1 = &ndims_tmp;
    $2 = &dimsizes_tmp;
    $3 = &data_tmp;
}
%typemap(argout, fragment="NumPy_Backward_Compatibility")
    (DIM_TYPE *NDIMS, DIM_TYPE **DIMSIZES, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_N_DIMS_ARRAY)
{
    npy_intp * dims = NULL;
    if( sizeof(npy_intp) == sizeof(DIM_TYPE) ) {
        dims = (npy_intp*)*$2;
    } else {
        dims = new npy_intp[*$1];
        for( size_t i=0; i<*$1; ++i ) {
            dims[i] = (npy_intp)(*$2)[i];
        }
    }
    PyObject* obj = PyArray_SimpleNewFromData(*$1, dims, DATA_TYPECODE, (void*)($3->get()));
    if( sizeof(npy_intp) != sizeof(DIM_TYPE) ) delete [] dims;
    if (!obj) SWIG_fail;
    set_sp_base((PyArrayObject*)obj, *$3);
    $result = SWIG_Python_AppendOutput($result, obj);
}


/* Typemap suite for (DIM_TYPE *NDIMS, DIM_TYPE **DIMSIZES, DATA_TYPE** ARGOUTVIEW_SP_DIM_ARRAY)
 */
%typemap(in,numinputs=0)
    (daal::services::Collection<size_t> const** DIMS, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_DIMS_ARRAY)
    (daal::services::Collection<size_t>* dims_tmp, daal::services::SharedPtr< DATA_TYPE > data_tmp)
{
    $1 = &dims_tmp;
    $2 = &data_tmp;
}
%typemap(argout, fragment="NumPy_Backward_Compatibility")
    (daal::services::Collection<size_t> const** DIMS, daal::services::SharedPtr< DATA_TYPE >* ARGOUTVIEW_SP_DIMS_ARRAY)
{
    npy_intp * dims = NULL;
    dims = new npy_intp[(*$1)->size()];
    for( size_t i=0; i<(*$1)->size(); ++i ) {
        dims[i] = (npy_intp)(*$1)->get(i);
    }
    PyObject* obj = PyArray_SimpleNewFromData((*$1)->size(), dims, DATA_TYPECODE, (void*)($2->get()));
    delete [] dims;
    if (!obj) SWIG_fail;
    set_sp_base((PyArrayObject*)obj, *$2);
    $result = SWIG_Python_AppendOutput($result, obj);
}

%enddef    /* %numpy_argout_typemaps() macro */
/* *************************************************************** */

/* Concrete instances of the %numpy_typemaps() macro: Each invocation
 * below applies all of the typemaps above to the specified data type.
 */
%numpy_in_typemaps    (unsigned char     , NPY_UBYTE    , size_t, FORCE_INPLACE)
%numpy_in_typemaps    (int               , NPY_INT      , size_t, FORCE_INPLACE)
%numpy_in_typemaps    (size_t            , NPY_UINT64   , size_t, FORCE_INPLACE)
%numpy_in_typemaps    (double            , NPY_DOUBLE   , size_t, FORCE_INPLACE)
%numpy_in_typemaps    (float             , NPY_FLOAT    , size_t, FORCE_INPLACE)
%numpy_in_typemaps    (unsigned char     , NPY_UBYTE    , size_t, ALLOW_CONVERSION)
%numpy_in_typemaps    (int               , NPY_INT      , size_t, ALLOW_CONVERSION)
%numpy_in_typemaps    (size_t            , NPY_UINT64   , size_t, ALLOW_CONVERSION)
%numpy_in_typemaps    (double            , NPY_DOUBLE   , size_t, ALLOW_CONVERSION)
%numpy_in_typemaps    (float             , NPY_FLOAT    , size_t, ALLOW_CONVERSION)
%numpy_argout_typemaps(unsigned char     , NPY_UBYTE    , size_t)
%numpy_argout_typemaps(int               , NPY_INT      , size_t)
%numpy_argout_typemaps(size_t            , NPY_UINT64   , size_t)
%numpy_argout_typemaps(double            , NPY_DOUBLE   , size_t)
%numpy_argout_typemaps(float             , NPY_FLOAT    , size_t)

/* Typemap suite for (void *PTR, int SSIZE, int NROWS, int * FEATUREOFFS, int * FEATURESIZES, int NFEATURES)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY,
           fragment="NumPy_Macros")
(PyArrayObject* SARRAY)
{
    $1 = is_array($input) && array_numdims($input) == 1 && array_descr($input) && array_descr($input)->names;
}
%typemap(in,
         fragment="NumPy_Fragments")
    (PyArrayObject* SARRAY)
    (PyObject* ary)
{
    $1 = obj_to_array_no_conversion($input, NPY_NOTYPE);
    if (!$1 || !require_dimensions($1,1) || !require_contiguous($1) || !require_native($1)) {
        SWIG_fail;
    }
}
/* ***************************************************************
 * The follow macro expansion does not work, because C++ bool is 4
 * bytes and NPY_BOOL is 1 byte
 *
 *    %numpy_typemaps(bool, NPY_BOOL, int)
 */

#endif /* SWIGPYTHON */
