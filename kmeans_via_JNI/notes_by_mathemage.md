# Notes on kmeans via JNI

## Settings on MacBook Pro 12.1
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

### Operating Systems
* Ubuntu* 17.04 (zesty)

### C/C++ Compilers
* gcc version 6.3.0 20170406 (Ubuntu 6.3.0-12ubuntu2)

### Java* Compilers:
```
$ java -version
openjdk version "1.8.0_131"
OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.17.04.1-b11)
OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)
```

## Installation

### Installing from the Sources

#### Installation Steps
1. Clone the sources from GitHub* as follows:
```
$ git clone --recursive https://github.com/01org/daal.git
Cloning into 'daal'...
remote: Counting objects: 20865, done.
remote: Total 20865 (delta 0), reused 0 (delta 0), pack-reused 20865
Receiving objects: 100% (20865/20865), 448.05 MiB | 1.68 MiB/s, done.
Resolving deltas: 100% (16684/16684), done.
Checking out files: 100% (5367/5367), done.
```

2. Build Intel DAAL via the command-line interface with the following commands:
 
 (on Linux\* using GNU Compiler Collection\*)

```
make daal PLAT=lnx32e COMPILER=gnu
```
            
 * JNI compilation issue:
 
    ```
    ========= Building threading =========
    ========= Building java =========
    g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp && printf '\n%s\n' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps):' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o: $(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps)' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.trigger = g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp' >> __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp && mv -f __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d
    lang_service/java/com/intel/daal/algorithms/parameter.cpp:18:17: fatal error: jni.h: No such file or directory
     #include <jni.h>
                     ^
    compilation terminated.
    makefile:576: recipe for target '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o' failed
    make: *** [__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o] Error 1
    ```
    
    Stack Overflow suggests this [solution](https://stackoverflow.com/questions/14529720/how-to-make-jni-h-be-found):

    ```commandline
    make -I/usr/lib/jvm/jdk*/include
    ```
    
    Hence, I added following line to [makefile](../makefile):
    
    ```
    $(THR.objs): INCLUDES += $(addprefix -I, /usr/lib/jvm/java-8-openjdk-amd64/include/)
    ```

 * `libdaal_core.a` compilation issue:
 
    ```bash
    ========= Building core =========
    printf "create __work_gnu/lnx32e/daal/lib/libdaal_core.a\naddlib externals/mklfpk/lnx/lib/intel64/libdaal_vmlipp_core.a\n addlib __work_gnu/lnx32e/kernel/libdaal_core_link.a\n\n\n\nsave\n" | ar -M && printf '\n%s\n' '__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps):' '__work_gnu/lnx32e/daal/lib/libdaal_core.a: $(__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps)' '__work_gnu/lnx32e/daal/lib/libdaal_core.a.trigger = printf "create __work_gnu/lnx32e/daal/lib/libdaal_core.a\naddlib externals/mklfpk/lnx/lib/intel64/libdaal_vmlipp_core.a\n addlib __work_gnu/lnx32e/kernel/libdaal_core_link.a\n\n\n\nsave\n" | ar -M' >> __work_gnu/lnx32e/daal/lib/libdaal_core.a.d.tmp && mv -f __work_gnu/lnx32e/daal/lib/libdaal_core.a.d.tmp __work_gnu/lnx32e/daal/lib/libdaal_core.a.d
    ar: __work_gnu/lnx32e/kernel/libdaal_core_link.a: No such file or directory
    makefile:356: recipe for target '__work_gnu/lnx32e/daal/lib/libdaal_core.a' failed
    make: *** [__work_gnu/lnx32e/daal/lib/libdaal_core.a] Error 9
    ```

    TODO to be solved
