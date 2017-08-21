# Notes on kmeans via JNI

## Settings on MacBook Pro 12.1
Intel DAAL supports the IA-32 and Intel(R) 64 architectures. For a detailed explanation of these architecture names, read the [Intel Architecture Platform Terminology for Development Tools](https://software.intel.com/en-us/articles/intel-architecture-platform-terminology-for-development-tools) article.

### Operating Systems
* Ubuntu* 17.04 (zesty)

### C/C++ Compilers
* gcc version 6.3.0 20170406 (Ubuntu 6.3.0-12ubuntu2)

### Java* Compilers:

    $ java -version
    openjdk version "1.8.0_131"
    OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.17.04.1-b11)
    OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)

## Installation

### Installing from the Sources

#### Installation Steps
1. Clone the sources from GitHub* as follows:

        $ git clone --recursive https://github.com/01org/daal.git
        Cloning into 'daal'...
        remote: Counting objects: 20865, done.
        remote: Total 20865 (delta 0), reused 0 (delta 0), pack-reused 20865
        Receiving objects: 100% (20865/20865), 448.05 MiB | 1.68 MiB/s, done.
        Resolving deltas: 100% (16684/16684), done.
        Checking out files: 100% (5367/5367), done.

2. Set an environment variable for one of the supported Java* compilers; for example:

        $ echo $PATH
        /home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin

        $ cat >>~/.bashrc
        # Java JDK -> $PATH
        JAVA_JDK=/usr/lib/jvm/java-8-openjdk-amd64/bin
        export JAVA_JDK
        PATH=$JAVA_JDK/bin:$PATH
        export PATH

        $ source ~/.bashrc

        $ echo $PATH
        /usr/lib/jvm/java-8-openjdk-amd64/bin/bin:/home/mathemage/torch/install/bin:jdk-install-dir/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/home/mathemage/torch/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin

3. Build Intel DAAL via the command-line interface with the following commands (on Linux using GNU Compiler Collection):

        make daal PLAT=lnx32e COMPILER=gnu
            
 * JNI compilation issue:
 
        ========= Building threading =========
        ========= Building java =========
        g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp && printf '\n%s\n' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps):' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o: $(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps)' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.trigger = g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp' >> __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp && mv -f __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d
        lang_service/java/com/intel/daal/algorithms/parameter.cpp:18:17: fatal error: jni.h: No such file or directory
         #include <jni.h>
                         ^
        compilation terminated.
        makefile:576: recipe for target '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o' failed
        make: *** [__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o] Error 1
    
    Stack Overflow suggests this [solution](https://stackoverflow.com/questions/14529720/how-to-make-jni-h-be-found):

        make -I/usr/lib/jvm/jdk*/include
    
    Hence, I added following line to [makefile](../makefile):
    
        $(THR.objs): INCLUDES += $(addprefix -I, /usr/lib/jvm/java-8-openjdk-amd64/include/)

 * `libdaal_core.a` compilation issue:
 
        ========= Building core =========
        printf "create __work_gnu/lnx32e/daal/lib/libdaal_core.a\naddlib externals/mklfpk/lnx/lib/intel64/libdaal_vmlipp_core.a\n addlib __work_gnu/lnx32e/kernel/libdaal_core_link.a\n\n\n\nsave\n" | ar -M && printf '\n%s\n' '__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps):' '__work_gnu/lnx32e/daal/lib/libdaal_core.a: $(__work_gnu/lnx32e/daal/lib/libdaal_core.a.mkdeps)' '__work_gnu/lnx32e/daal/lib/libdaal_core.a.trigger = printf "create __work_gnu/lnx32e/daal/lib/libdaal_core.a\naddlib externals/mklfpk/lnx/lib/intel64/libdaal_vmlipp_core.a\n addlib __work_gnu/lnx32e/kernel/libdaal_core_link.a\n\n\n\nsave\n" | ar -M' >> __work_gnu/lnx32e/daal/lib/libdaal_core.a.d.tmp && mv -f __work_gnu/lnx32e/daal/lib/libdaal_core.a.d.tmp __work_gnu/lnx32e/daal/lib/libdaal_core.a.d
        ar: __work_gnu/lnx32e/kernel/libdaal_core_link.a: No such file or directory
        makefile:356: recipe for target '__work_gnu/lnx32e/daal/lib/libdaal_core.a' failed
        make: *** [__work_gnu/lnx32e/daal/lib/libdaal_core.a] Error 9

    I make-cleaned and reinstall again. This time, I got a different issue (see below).
    
 * JNI issue again:
 
        $ make _daal_jj PLAT=lnx32e COMPILER=gnu
        ========= Building java =========
        g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp && printf '\n%s\n' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps += makefile build/cmplr.gnu.mk build/common.mk build/deps.mk makefile.ver' '$(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps):' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o: $(__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.mkdeps)' '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.trigger = g++ -D__int64="long long" -D__int32="int" -m64 -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2 -fstack-protector -fPIC -std=c++11   -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED @./__work_gnu/lnx32e/jni_tmpdir/inc_j_folders.txt  -c  -o__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o lang_service/java/com/intel/daal/algorithms/parameter.cpp' >> __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp && mv -f __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d.tmp __work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o.d
        lang_service/java/com/intel/daal/algorithms/parameter.cpp:18:17: fatal error: jni.h: No such file or directory
         #include <jni.h>
                         ^
        compilation terminated.
        makefile:577: recipe for target '__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o' failed
        make: *** [__work_gnu/lnx32e/jni_tmpdir/com/intel/daal/algorithms/parameter.o] Error 1
 
    The [solution](https://github.com/01org/daal/issues/4) is taken from the issue #4 of DAAL's GitHub, which finally compiles:
    
        $ make daal PLAT=lnx32e COMPILER=gnu CPATH=/usr/lib/jvm/java-8-openjdk-amd64/include:/usr/lib/jvm/java-8-openjdk-amd64/include/linux/
        ========= Building core =========
        ========= Building threading =========
        ========= Building java =========
        ========= Building release =========
        make: Nothing to be done for 'daal'.
        
    or, if you have installed Oracle JDK:
    
        $ make daal PLAT=lnx32e COMPILER=gnu CPATH=/usr/lib/jvm/java-8-oracle/include:/usr/lib/jvm/java-8-oracle/include/linux/
    
  ## DAAL via Java
  
  ### Build
  The [Intel DAAL Getting Started](https://software.intel.com/en-us/get-started-with-daal-for-linux) page mentions using DAAL via Java:
   
  > To build and run Java code examples, use the version of the Java Virtual Machine* corresponding to the architecture parameter you provided to the daalvars.sh script during setting environment variables.
  > 
  > 1. Free 4 gigabytes of memory on your system.
  > 2. Build examples:
  >
  > Go to the Java examples directory and execute the launcher command with the build parameter:
  > 
  >     cd <install dir>/daal/examples/java
  >     launcher.sh build $PATH_TO_JAVAC
  >
  > The command builds executables `*.class` (for example, `CholeskyBatch.class`) in the
  > `daal/examples/java/com/intel/daal/examples/<example name>` directory.
  > 3. Run examples:
  >
  > Go to the Java examples directory and execute the launcher command with the run parameter:
  >
  >     cd <install dir>/daal/examples/java
  >     launcher.sh {ia32|intel64} run $PATH_TO_JAVAC
  >
  > Choose the same architecture parameter as you provided to the `daalvars.sh` script.
  > The output for each example is written to the file `<example name>.res` located in the `./_results/ia32 or ./_results/intel64` directory, depending on the specified architecture.

  
  ### Examples by Intel
  There are some prepared examples of DAAL in Java by Intel, e.g., *kmeans-clustering* examples in [h2oai/daal/examples/java/com/intel/daal/examples/kmeans/](../examples/java/com/intel/daal/examples/kmeans/)
