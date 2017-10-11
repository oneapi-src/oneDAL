/* --------------------------------------------------------------------
 * The original version of this file is part of SWIG, which is
 * licensed as a whole under version 3 (or any later version) of the
 * GNU General Public License. Some additional terms also apply to
 * certain portions of SWIG. The full details of the SWIG license and
 * copyrights can be found in the LICENSE and COPYRIGHT files included
 * with the SWIG source code as distributed by the SWIG developers and
 * at http://www.swig.org/legal.html.
 * --------------------------------------------------------------------- */

// This is a helper file for SharedPtr and should not be included directly.

// The main implementation detail in using this smart pointer of a type is to customise the code generated
// to use a pointer to the smart pointer of the type, rather than the usual pointer to the underlying type.
// So for some type T, SharedPtr<T> * is used rather than T *.

// FS: This file is derived from the original SWIG dall_shared_ptr.i.
//     It has been altered to work with DAAL's SharedPtr.
//     "shared_ptr" was replaced with "SharedPtr" all over the file.
//     Other changes are marked with FS:

// FS: The original file was designed so that namespaces could be boost or std or std::tr1
//     DAAl's SharedPtr is on the 3rd level of namespace hierarchy (daal::services::interface1)
//     -> added subsubnamespace
//     For example for std::tr1::if11, use:
// #define SWIG_SHAREDPTR_NAMESPACE std
// #define SWIG_SHAREDPTR_SUBNAMESPACE tr1
// #define SWIG_SHAREDPTR_SUBSUBNAMESPACE if11

// FS: default is daal::services::interface1
#ifndef SWIG_SHAREDPTR_NAMESPACE
# define SWIG_SHAREDPTR_NAMESPACE daal
#endif
#ifndef SWIG_SHAREDPTR_SUBNAMESPACE
# define SWIG_SHAREDPTR_SUBNAMESPACE services
#endif
/* #ifndef SWIG_SHAREDPTR_SUBSUBNAMESPACE */
/* # define SWIG_SHAREDPTR_SUBSUBNAMESPACE interface1 */
/* #endif */

#if defined(SWIG_SHAREDPTR_SUBNAMESPACE)
# if defined(SWIG_SHAREDPTR_SUBSUBNAMESPACE)
#  define SWIG_SHAREDPTR_QNAMESPACE SWIG_SHAREDPTR_NAMESPACE::SWIG_SHAREDPTR_SUBNAMESPACE::SWIG_SHAREDPTR_SUBSUBNAMESPACE
# else
#  define SWIG_SHAREDPTR_QNAMESPACE SWIG_SHAREDPTR_NAMESPACE::SWIG_SHAREDPTR_SUBNAMESPACE
# endif
#else
# define SWIG_SHAREDPTR_QNAMESPACE SWIG_SHAREDPTR_NAMESPACE
#endif

namespace SWIG_SHAREDPTR_NAMESPACE {
#if defined(SWIG_SHAREDPTR_SUBNAMESPACE)
  namespace SWIG_SHAREDPTR_SUBNAMESPACE {
# if defined(SWIG_SHAREDPTR_SUBSUBNAMESPACE)
    namespace SWIG_SHAREDPTR_SUBSUBNAMESPACE {
# endif
#endif
      // FS: just a prototype, not a realy class, we might %include the c++ file
      template <class T> class SharedPtr;
#if defined(SWIG_SHAREDPTR_SUBNAMESPACE)
# if defined(SWIG_SHAREDPTR_SUBSUBNAMESPACE)
    }
# endif
  }
#endif
}
// FS: no more mentioning of subsubnamespace from here on

%fragment("SWIG_null_deleter", "header") {
// FS: Intel(R) DAAL has its own deleter iface
struct SWIG_null_deleter : public daal::services::DeleterIface
{
    void operator() (void *) const {}

    // FS: from daal::services::DeleterIface
    virtual void operator() (const void *ptr) {}
};
%#define SWIG_NO_NULL_DELETER_0 , SWIG_null_deleter()
%#define SWIG_NO_NULL_DELETER_1
%#define SWIG_NO_NULL_DELETER_SWIG_POINTER_NEW
%#define SWIG_NO_NULL_DELETER_SWIG_POINTER_OWN
}


// Workaround empty first macro argument bug
#define SWIGEMPTYHACK
// Main user macro for defining shared_ptr typemaps for both const and non-const pointer types
%define %shared_ptr(TYPE...)
%feature("smartptr", noblock=1) TYPE { SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > }
SWIG_SHAREDPTR_TYPEMAPS(SWIGEMPTYHACK, TYPE)
SWIG_SHAREDPTR_TYPEMAPS(const, TYPE)
%enddef

// Legacy macros
%define SWIG_SHAREDPTR(PROXYCLASS, TYPE...)
#warning "SWIG_SHAREDPTR(PROXYCLASS, TYPE) is deprecated. Please use %shared_ptr(TYPE) instead."
%shared_ptr(TYPE)
%enddef

%define SWIG_SHAREDPTR_DERIVED(PROXYCLASS, BASECLASSTYPE, TYPE...)
#warning "SWIG_SHAREDPTR_DERIVED(PROXYCLASS, BASECLASSTYPE, TYPE) is deprecated. Please use %shared_ptr(TYPE) instead."
%shared_ptr(TYPE)
%enddef


// Set SHAREDPTR_DISOWN to $disown if required, for example
// #define SHAREDPTR_DISOWN $disown
#if !defined(SHAREDPTR_DISOWN)
#define SHAREDPTR_DISOWN 0
#endif

%fragment("SWIG_null_deleter_python", "header", fragment="SWIG_null_deleter") {
%#define SWIG_NO_NULL_DELETER_SWIG_BUILTIN_INIT
}

// Language specific macro implementing all the customisations for handling the smart pointer
%define SWIG_SHAREDPTR_TYPEMAPS(CONST, TYPE...)

// %naturalvar is as documented for member variables
%naturalvar TYPE;
%naturalvar SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >;

// destructor wrapper customisation
%feature("unref") TYPE "delete smartarg1;"

//"if (debug_shared) { cout << \"deleting use_count: \" << (*smartarg1).use_count() << \" [\" << (boost::get_deleter<SWIG_null_deleter>(*smartarg1) ? std::string(\"CANNOT BE DETERMINED SAFELY\") : ( (*smartarg1).get() ? (*smartarg1)->getValue() : std::string(\"NULL PTR\") )) << \"]\" << endl << flush; }\n"

//"(void)arg1; decrement_py_ref_and_delete(smartarg1);"

// Typemap customisations...

// plain value
%typemap(in) CONST TYPE (void *argp, int res = 0) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (!argp) {
    %argument_nullref("$type", $symname, $argnum);
  } else {
    $1 = *(%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *)->get());
    if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
  }
}
%typemap(out) CONST TYPE {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(new $1_ltype(($1_ltype &)$1));
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

%typemap(varin) CONST TYPE {
  void *argp = 0;
  int newmem = 0;
  int res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %variable_fail(res, "$type", "$name");
  }
  if (!argp) {
    %argument_nullref("$type", $symname, $argnum);
  } else {
    $1 = *(%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *)->get());
    if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
  }
}
%typemap(varout) CONST TYPE {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(new $1_ltype(($1_ltype &)$1));
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

// plain pointer
// Note: $disown not implemented by default as it will lead to a memory leak of the SharedPtr instance
%typemap(in) CONST TYPE * (void  *argp = 0, int res = 0, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > tempshared, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartarg = 0) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SHAREDPTR_DISOWN | %convertptr_flags, &newmem);
  if ((!SWIG_IsOK(res) || !argp) && $input != Py_None) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    tempshared = *%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    $1 = %const_cast(tempshared.get(), $1_ltype);
  } else {
    smartarg = %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    $1 = %const_cast((smartarg ? smartarg->get() : 0), $1_ltype);
  }
  if(!$1 && $input != Py_None) %argument_fail(res, "$type", $symname, $argnum); 
}

%typemap(out, fragment="SWIG_null_deleter_python") CONST TYPE * {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = $1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_$owner) : 0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), $owner | SWIG_POINTER_OWN));
}

%typemap(varin) CONST TYPE * {
  void *argp = 0;
  int newmem = 0;
  int res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res) || !argp) {
    %variable_fail(res, "$type", "$name");
  }
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > tempshared;
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartarg = 0;
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    tempshared = *%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    $1 = %const_cast(tempshared.get(), $1_ltype);
  } else {
    smartarg = %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    $1 = %const_cast((smartarg ? smartarg->get() : 0), $1_ltype);
  }
  if(!$1) %argument_fail(res, "$type", $symname, $argnum); 
}
%typemap(varout, fragment="SWIG_null_deleter_python") CONST TYPE * {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = $1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_0) : 0;
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

// plain reference
%typemap(in) CONST TYPE & (void  *argp = 0, int res = 0, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > tempshared) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (!argp) { %argument_nullref("$type", $symname, $argnum); }
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    tempshared = *%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    $1 = %const_cast(tempshared.get(), $1_ltype);
  } else {
    $1 = %const_cast(%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *)->get(), $1_ltype);
  }
  if(!$1) %argument_fail(res, "$type", $symname, $argnum); 
}
%typemap(out, fragment="SWIG_null_deleter_python") CONST TYPE & {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_$owner);
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

%typemap(varin) CONST TYPE & {
  void *argp = 0;
  int newmem = 0;
  int res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %variable_fail(res, "$type", "$name");
  }
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > tempshared;
  if (!argp) { %argument_nullref("$type", $symname, $argnum); }
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    tempshared = *%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    $1 = *%const_cast(tempshared.get(), $1_ltype);
  } else {
    $1 = *%const_cast(%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *)->get(), $1_ltype);
  }
  if(!$1) %argument_fail(res, "$type", $symname, $argnum); 
}
%typemap(varout, fragment="SWIG_null_deleter_python") CONST TYPE & {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(&$1 SWIG_NO_NULL_DELETER_0);
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

// plain pointer by reference
// Note: $disown not implemented by default as it will lead to a memory leak of the SharedPtr instance
%typemap(in) TYPE *CONST& (void  *argp = 0, int res = 0, $*1_ltype temp = 0, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > tempshared) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SHAREDPTR_DISOWN | %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res) || !argp) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    tempshared = *%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    delete %reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *);
    temp = %const_cast(tempshared.get(), $*1_ltype);
  } else {
    temp = %const_cast(%reinterpret_cast(argp, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *)->get(), $*1_ltype);
  }
  if(!temp) %argument_fail(res, "$type", $symname, $argnum); 
  $1 = &temp;
}
%typemap(out, fragment="SWIG_null_deleter_python") TYPE *CONST& {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(*$1 SWIG_NO_NULL_DELETER_$owner);
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

%typemap(varin) TYPE *CONST& %{
#error "varin typemap not implemented"
%}
%typemap(varout) TYPE *CONST& %{
#error "varout typemap not implemented"
%}

// SharedPtr by value
%typemap(in) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > (void *argp, int res = 0) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (argp) $1 = *(%reinterpret_cast(argp, $&ltype));
  if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, $&ltype);
}
%typemap(out) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = $1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >($1) : 0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

%typemap(varin) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > {
  int newmem = 0;
  void *argp = 0;
  int res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %variable_fail(res, "$type", "$name");
  }
  $1 = argp ? *(%reinterpret_cast(argp, $&ltype)) : SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE >();
  if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, $&ltype);
}
%typemap(varout) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = $1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >($1) : 0;
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

// SharedPtr by reference
%typemap(in) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > & (void *argp, int res = 0, $*1_ltype tempshared) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    if (argp) tempshared = *%reinterpret_cast(argp, $ltype);
    delete %reinterpret_cast(argp, $ltype);
    $1 = &tempshared;
  } else {
    $1 = (argp) ? %reinterpret_cast(argp, $ltype) : &tempshared;
  }
}
%typemap(out) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > & {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = *$1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(*$1) : 0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

%typemap(varin) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > & %{
#error "varin typemap not implemented"
%}
%typemap(varout) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > & %{
#error "varout typemap not implemented"
%}

// SharedPtr by pointer
%typemap(in) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > * (void *argp, int res = 0, $*1_ltype tempshared) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (newmem & SWIG_CAST_NEW_MEMORY) {
    if (argp) tempshared = *%reinterpret_cast(argp, $ltype);
    delete %reinterpret_cast(argp, $ltype);
    $1 = &tempshared;
  } else {
    $1 = (argp) ? %reinterpret_cast(argp, $ltype) : &tempshared;
  }
}
%typemap(out) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > * {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = $1 && *$1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(*$1) : 0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
  if ($owner) delete $1;
}

%typemap(varin) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > * %{
#error "varin typemap not implemented"
%}
%typemap(varout) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > * %{
#error "varout typemap not implemented"
%}

// SharedPtr by pointer reference
%typemap(in) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *& (void *argp, int res = 0, SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > tempshared, $*1_ltype temp = 0) {
  int newmem = 0;
  res = SWIG_ConvertPtrAndOwn($input, &argp, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), %convertptr_flags, &newmem);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res, "$type", $symname, $argnum); 
  }
  if (argp) tempshared = *%reinterpret_cast(argp, $*ltype);
  if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, $*ltype);
  temp = &tempshared;
  $1 = &temp;
}
%typemap(out) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *& {
  SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *smartresult = *$1 && **$1 ? new SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >(**$1) : 0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), SWIG_POINTER_OWN));
}

%typemap(varin) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *& %{
#error "varin typemap not implemented"
%}
%typemap(varout) SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *& %{
#error "varout typemap not implemented"
%}

// Typecheck typemaps
// Note: SWIG_ConvertPtr with void ** parameter set to 0 instead of using SWIG_ConvertPtrAndOwn, so that the casting 
// function is not called thereby avoiding a possible smart pointer copy constructor call when casting up the inheritance chain.
%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER,noblock=1) 
                      TYPE CONST,
                      TYPE CONST &,
                      TYPE CONST *,
                      TYPE *CONST&,
                      SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >,
                      SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > &,
                      SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *,
                      SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE > *& {
  int res = SWIG_ConvertPtr($input, 0, $descriptor(SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< TYPE > *), 0);
  $1 = SWIG_CheckState(res);
}


// various missing typemaps - If ever used (unlikely) ensure compilation error rather than runtime bug
%typemap(in) CONST TYPE[], CONST TYPE[ANY], CONST TYPE (CLASS::*) %{
#error "typemaps for $1_type not available"
%}
%typemap(out) CONST TYPE[], CONST TYPE[ANY], CONST TYPE (CLASS::*) %{
#error "typemaps for $1_type not available"
%}


%template() SWIG_SHAREDPTR_QNAMESPACE::SharedPtr< CONST TYPE >;


%enddef
