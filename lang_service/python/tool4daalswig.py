#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

description = """
A tool to help automating tasks like creating skeleton templates and running tests
and verifying their output.
See pare.py for details about C++ parsing.
See swig_interface.py for details about generating a SWIG interface template file.
"""

import os
import re
import subprocess
import sys
import shutil
from collections import defaultdict
from swig_interface import swig_interface
from os.path import join as jp
from shutil import copyfile

IS_LIN = IS_WIN = IS_MAC = False
if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
else:
    IS_WIN = True
    import ctypes
    SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
    ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
    CREATE_NO_WINDOW = 0x08000000    # From Windows API
    subprocess_flags = CREATE_NO_WINDOW


def extract_spark_results(res_file):
    with open(res_file, 'r') as f:
        lines = f.readlines()

    regex = r'^([-+]?\d+\.?\d*\s+)+'
    matched = []
    for line in lines:
        match = re.search(regex, line)
        if match:
            matched.append(line.replace('+', '').replace('\t', ''))
    return matched


def compare_results(sample, result, expected):
    for line1, line2 in zip(result, expected):
        try:
            assert line1.replace(' ', '').rstrip('\n') == line2.replace(' ', '').rstrip('\n')
        except AssertionError:
            print("{} failed.  Lines don't match:".format(sample))
            print(line1)
            print(line2)
            print('#Failure#')


def delete_spark_results():
    if os.path.exists('_results'):
        shutil.rmtree('_results')


def get_pyspark_results(res_file):
    with open(res_file, 'r') as f:
        res = [line for line in f.readlines() if re.match(r'^[\d\-\s]', line) and line != '\n']
    return res


def array_string(l):
    ret = '['
    for x in l:
        ret += "\n        '" + x + "',"
    return ret + '\n    ]'

def templates_string(td):
    ret = ''
    for x in td:
        ret += '\n' + ' '*8 + "'" + x[0] + "': " + str(x[1])
    return ret


if __name__ == "__main__":
    import argparse
    default_ignore_tests = [
        # Please keep one bug per line, like
        # 'none',  # SAT-???
    ]
    if IS_WIN:
        default_ignore_tests.append('svm_multi_class_quality_metric_set_batch')

    # some examples still have differences to C++ which are ok
    # here we provide a filter that filters out lines that are allowed to differ
    # each example can provide its own, by default we only filter out empty lines
    ignoreoutput = {
        'compression_batch'  : lambda x: False if ' data checksum:' in x else True,
        'compression_online' : lambda x: False if ' data checksum:' in x else True,
        'compressor'         : lambda x: False if ' data checksum:' in x else True,
        'implicit_als_csr_distributed': lambda x: False if '19, 13, -7.74265e-0' in x else True,
    }

    checkoutroot = jp(os.path.dirname(os.path.abspath(__file__)), '..', '..')

    include_directories = ['algorithms', 'data_management', ]
    argParser = argparse.ArgumentParser(prog="tool4daalswig.py",
                                        description=description,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument('-c', '--create_templates', default=None, nargs='?', const='.',            help="creates new template files in given dir, compare with existing files")
    argParser.add_argument('-t', '--test',             default=[], nargs='?', const='all',            help="executes tests and compares with DAAL C++ test output. Accepts ','-separated list of 'all', 'examples', 'mpi', 'spark', 'mysql' and/or 'hadoop'")
    argParser.add_argument('-n', '--name',             default='all',                                     help="name of test to run")
    argParser.add_argument('-i', '--ignore',           default=None,                                      help="comma-separated list of to-be-ignored tests")
    argParser.add_argument('--env',                    default='all', choices=['all','py','cpp','pdb'],   help="env for tests to run")
    argParser.add_argument('--show-tests', '--show',   default=False, action='store_true',                help="prints all Python test names from daal/examples")
    argParser.add_argument('-v', '--verbose',          default=False, action='store_true',                help="prints complete output of tests")
    argParser.add_argument('-x', '--doc_test',         default=False, action='store_true',                help="run documentation test")
    argParser.add_argument('--serialize-tests',     default='true', choices=['true', 'false', 'keep'], help="skips serialization tests on all examples")
    argParser.add_argument('-g', '--gaps',             default=False, action='store_true',                help="find cpp tests not available for python")
    argParser.add_argument('-d', '--output-dir',       default=jp(os.getcwd(), 'test_output'),            help="store test logfile in given directory")
    argParser.add_argument('--example-search-dir',     default=jp(os.environ['DAALROOT'], 'examples', 'python'),    help="search python examples in given dir")
    argParser.add_argument('--sample-search-dir',      default=jp(checkoutroot, 'samples', 'python'),     help="search python samples in given dir")
    argParser.add_argument('-w', '--work-dir',         default=jp(os.getcwd(), 'tmp_testing'),            help="work dir for testing")

    args = argParser.parse_args()
    args.ignore = default_ignore_tests if args.ignore is None else args.ignore.split(',')
    if 'all' in args.test:
        args.test = 'examples,mpi,mysql,hadoop,spark'

    # Used later to note errors that came up (e.g. header files and namespaces are outliers) and writes them to <current_directory>/error.txt
    error_collection_string = ""
    error_template_string = ""

    from time import gmtime, strftime
    current_time = strftime("%Y-%m-%d_%H%M%S", gmtime())
    oldcwd = os.path.abspath('.')
    toolscript_loc = os.path.dirname(os.path.abspath(__file__))


    if args.show_tests:
        os.chdir(jp(toolscript_loc, '..', '..', 'examples'))
        proc = subprocess.Popen(["ls -R | grep .py$"], stdout=subprocess.PIPE, shell=True)
        (test_list, test_list_err) = proc.communicate()
        print("These are all Python tests that exist in daal/examples: \n" + test_list.decode('ascii'))
        os.chdir(oldcwd)
        exit()

    # This will not ever overwrite existing templates, only create non-existing ones
    if args.create_templates or args.gaps:
        iface = swig_interface(jp(os.environ['DAALROOT'], 'include'))
        iface.read()
    if args.create_templates:
        iface.digest()
        # We are ready to dump our data!
        iface.write_and_compare(args.create_templates, toolscript_loc)

    if args.gaps:
        test_directory = os.path.abspath(jp(toolscript_loc, '..', '..', 'examples'))
        samples_directory = os.path.abspath(jp(toolscript_loc, '..', '..', 'samples'))
        directories = [test_directory, samples_directory]
        for cdir in directories:
            for (dirpath, dirnames, filenames) in os.walk(jp(cdir, 'cpp')):
                for cppfile in filenames:
                    if cppfile.endswith('.cpp'):
                        pyfile = jp(dirpath.replace('cpp', 'python'), cppfile).replace('.cpp', '.py')
                        if not os.path.isfile(pyfile):
                            print(pyfile)

    if args.test:
        test_output_dir = args.output_dir
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        test_output_file = jp(test_output_dir, "test__" + args.name.replace(',', '_')[:32] + '__' + args.env + "__" + current_time + ".txt")
        PYCMD = sys.executable
        with open(test_output_file, 'a') as output_file:
            print("##########################################################")
            print("# Running tool4daalswig.py\n#")
            print("# Ignore: " + ', '.join(args.ignore))
            output_file.write("##########################################################\n")
            output_file.write("# Running tool4daalswig.py\n#\n")
            output_file.write("# Ignore: " + ', '.join(args.ignore) + '\n')
            tests_to_ignore = args.ignore
            py_success_number = 0
            py_failure_number = 0
            cpp_success_number = 0
            cpp_failure_number = 0
            java_success_number = 0
            differences_success_number = 0
            differences_failure_number = 0
            total_tests = 0
            py_cpp_test_passes = 0

            test_directory = args.example_search_dir
            samples_directory = args.sample_search_dir

            py_examples_location = jp(args.work_dir, 'examples', 'python')
            py_samples_location = jp(args.work_dir, 'samples', 'python')
            cpp_examples_location = jp(args.work_dir, 'examples', 'cpp')
            cpp_samples_location = jp(args.work_dir, 'samples', 'cpp')
            java_samples_location = jp(args.work_dir, 'samples', 'java')
            examples_data_location = jp(args.work_dir, 'examples', 'data')

            directories = []
            if 'examples' in args.test:
                directories.append(py_examples_location)
            if any(s in args.test for s in ['mpi', 'mysql']):
                directories.append(py_samples_location)

            shutil.rmtree(examples_data_location, True)
            shutil.copytree(jp(os.environ['DAALROOT'], 'examples', 'data'), examples_data_location)

            if args.env in ['py', 'all',]:
                # let's first delete old stuff
                shutil.rmtree(py_examples_location, True)
                shutil.rmtree(py_samples_location, True)
                # now copy the examples and samples into our dir-structure
                shutil.copytree(args.example_search_dir, py_examples_location)
                shutil.copytree(args.sample_search_dir, py_samples_location)

            if 'spark' in args.test:
                # Remove old files
                shutil.rmtree(java_samples_location, True)

                # Copy samples into our directory structure
                shutil.copytree(jp(samples_directory, '..', 'java'), java_samples_location)

            if args.env in ['cpp', 'all',]:
                # we have to copy the cpp examples, we can't work in the install-dir
                # let's first delete old stuff
                shutil.rmtree(cpp_examples_location, True)
                shutil.rmtree(cpp_samples_location, True)
                # now copy the examples and samples into our dir-structure
                shutil.copytree(jp(os.environ['DAALROOT'], 'examples', 'cpp'), cpp_examples_location)
                shutil.copytree(jp(os.environ['DAALROOT'], '..', 'samples', 'cpp'), cpp_samples_location)

                # DAAL have hardcoded the relative path location of the libraries onto .vcxproj files. Going through each one and replacing relevant lines
                if IS_WIN:
                    print("\nProcessing .vcxproj files because DAAL hardcoded relative paths to libraries...\n")
                    for root, subDir, files in os.walk(cpp_examples_location):
                        for f in files:
                            if ".vcxproj" in f:
                                vcxproj_path = os.path.abspath(jp(root, f))
                                with open(vcxproj_path, 'r') as old_vcxproj:
                                    old_vcxproj_data = old_vcxproj.readlines()

                                os.remove(vcxproj_path)
                                with open(vcxproj_path, 'w') as new_vcxproj:
                                    for old_line in old_vcxproj_data:
                                        if "<LibraryPath>" in old_line:
                                            if "intel64_win" in old_line:
                                                daal_lib_dir = jp(os.environ['DAALROOT'], "lib", "intel64_win") + ";"
                                                tbb_lib_dir = jp(os.environ['DAALROOT'], "..", "tbb", "lib", "intel64_win", "vc_mt") + ";"
                                                compiler_lib_dir = jp(os.environ['DAALROOT'], "..", "compiler", "lib", "intel64_win") + ";"
                                                new_line = "    <LibraryPath>" + daal_lib_dir + tbb_lib_dir + compiler_lib_dir + "$(LibraryPath)</LibraryPath>\n"
                                            else:
                                                daal_lib_dir = jp(os.environ['DAALROOT'], "lib", "ia32_win") + ";"
                                                tbb_lib_dir = jp(os.environ['DAALROOT'], "..", "tbb", "lib", "ia32_win", "vc_mt") + ";"
                                                compiler_lib_dir = jp(os.environ['DAALROOT'], "..", "compiler", "lib", "ia32_win") + ";"
                                                new_line = "    <LibraryPath>" + daal_lib_dir + tbb_lib_dir + compiler_lib_dir + "$(LibraryPath)</LibraryPath>\n"
                                        elif "<AdditionalIncludeDirectories>" in old_line:
                                            daal_include_dir = jp(os.environ['DAALROOT'], "include") + ";"
                                            new_line = "<AdditionalIncludeDirectories>" + daal_include_dir + r"..\source\utils;</AdditionalIncludeDirectories>"
                                        else:
                                            new_line = old_line
                                        new_vcxproj.write(str(new_line))


            # Go through the process once for 'examples,' and once for 'samples.'
            for cdir in directories:
                # print("All output has been saved in " + jp(toolscript_loc, "test_output", args.name + "_test_output_" + current_time + ".txt\n\n"))
                os.chdir(cdir)
                run_examples = True if 'examples' in cdir else False

                # see if we are supposed to run test in this dir
                if run_examples:
                    if 'examples' not in args.test:
                        continue
                elif not any(s in args.test for s in ['mpi', 'spark', 'hadoop', 'mysql']):
                    continue

                # See if we should run all tests in a directory
                #if run_examples:
                #    intermediate_dirs = jp('python', 'source')
                #else:
                #    intermediate_dirs = jp('python', 'mpi', 'sources')

                # if os.path.isdir(jp(cdir, intermediate_dirs, args.name)):
                #     tests_to_run = os.listdir(jp(cdir, intermediate_dirs, args.name))
                #     tests_to_run = [x.rstrip('.py') for x in tests_to_run]
                # elif args.name == 'mpi':
                #     tests_to_run = os.listdir(jp(cdir, intermediate_dirs))
                #     tests_to_run = [x.rstrip('.py') for x in tests_to_run]
                # else:
                #     tests_to_run = args.name.split(',')

                tests_to_run = args.name.split(',')

                # Goes through every directory to execute test script if args.name is 'all'. Otherwise, just finds the test you asked for
                # i.e. execute svd_online (which will execute both svd_online.py and C++ object file equivalent)
                for (dirpath, dirnames, filenames) in os.walk(cdir, topdown=True):
                    dirnames[:] = [d for d in dirnames if d not in ['spark', 'hadoop']]
                    for filename in filenames:
                        if filename.startswith('run_e') or filename.startswith('_') or not filename.endswith('.py'):
                            continue

                        filename_no_ext = filename.replace('.py', '')
                        ignore = filename_no_ext in tests_to_ignore
                        run_test = filename_no_ext in tests_to_run or args.name == 'all'
                        if not run_test or ignore or (os.path.basename(os.path.dirname(dirpath)) not in args.test and not run_examples):
                            continue
                        cur_dir = os.path.abspath('.')
                        total_tests += 1

                        error_occurred = False
                        # Extracts output from running python.py and C++ object file
                        py_output = ''
                        py_err = ''

                        full_filename = jp(dirpath, filename)
                        tmp_filename = full_filename

                        if args.serialize_tests != 'false' and run_examples and 'neural_networks' not in full_filename:
                            # Modify tests to serialize and deserialize Result objects
                            # Serialization of Input objects currently not supported by DAAL
                            # 1. looks for statements that instantiate an algorithm and get name of variable
                            # 2. finds comments that provide special result type and its package
                            #    <result-type> class from <package-name>
                            #    <result-type> must end with 'esult'
                            # 3. find compute statement, get name of result varialble
                            # 4. infer result type if not found in 2.
                            #    assumes package to look in is given in first import statemen
                            # 5. replace compute statement with code that
                            #    a) computes
                            #    b) serializes result
                            #    c) stores it an byte-buffer
                            #    d) de-serializes byte-buffer into new/empty result object
                            # 6. following code will use new result object

                            # infer namespace/package name from first import statement
                            def get_ns(var_name):
                                ns_match = re.search(r'daal\.algorithms\.([^ \n]+)', open(jp(dirpath, filename)).read())
                                return "daal.algorithms." + ns_match.group(1)

                            def add_serialization_code(orig_line, var_name, package, result):
                                leading_spaces = len(orig_line) - len(orig_line.lstrip())
                                indent = ' ' * leading_spaces
                                var_name = var_name.lstrip()
                                importresult = ''
                                if package == '':
                                    importresult = indent + "from " + get_ns(var_name) + " import " + result + '\n'
                                serialization_code = (
                                    indent + "import numpy as np\n"
                                    + indent + "from daal.data_management import InputDataArchive, OutputDataArchive\n"
                                    + importresult
                                    + indent + "inArch = InputDataArchive()\n"
                                    + indent + var_name + ".serialize(inArch)\n"
                                    + indent + "buffer = np.zeros(inArch.getSizeOfArchive(), dtype=np.ubyte)\n"
                                    + indent + "inArch.copyArchiveToArray(buffer)\n"
                                    + indent + "outArch = OutputDataArchive(buffer)\n"
                                    + indent + "newRes = " + package + result + '()\n'
                                    + indent + "newRes.deserialize(outArch)\n"
                                    + indent + var_name + " = newRes\n"
                                )
                                return orig_line + serialization_code

                            # Copy the file so we can modify one version
                            tmp_filename = jp(dirpath, '_' + filename)
                            data = open(full_filename).readlines()

                            with open(tmp_filename, 'w') as f:
                                result = package = algo = ''
                                # Replace data contents with new serialization code
                                for line in data:
                                    # identify algorithm variable and package name
                                    match_package = re.search(r'[aA]lgorithm = ([\w.]+\.)(Batch|Distributed|Online)', line)
                                    if match_package:
                                        package = match_package.group(1)
                                        algo = match_package.group(2)
                                    else:
                                        # identify comment providing result type and its package
                                        match_hint = re.search(r' \(?(\w+esult\w*) class from ([\w.]+)', line)
                                        if match_hint:
                                            result = match_hint.group(1)
                                            package = match_hint.group(2) + '.'
                                        else:
                                            # identify compute statement, if not provided infer result-type
                                            match_result = re.search(r'(.*) = (.*)\.(finalizeC|c)ompute', line)
                                            if match_result:
                                                res_name = match_result.group(1)
                                                if result == '':
                                                    if 'batch' in filename or 'finalizeC' in match_result.group(3) or algo.startswith('Batch'):
                                                        result = 'Result'
                                                    else:
                                                        result = 'PartialResult'
                                                # finally do the code replacement
                                                line = line.replace(line, add_serialization_code(line, res_name, package, result))
                                                # reset matches so we do not re-use next time
                                                result = package = algo = ''
                                    f.write(line)

                        print("##########################################################\n" + filename_no_ext)
                        if args.env in ['py', 'all', 'pdb']:
                            if run_examples:
                                python_command = [PYCMD, tmp_filename]
                            else:
                                python_command = ['mpirun', '-np', '4', PYCMD, tmp_filename]
                            try:
                                py_err = ''
                                if IS_WIN:
                                    py_output = subprocess.check_output(python_command, stderr=subprocess.STDOUT, creationflags=subprocess_flags).decode('ascii')
                                else:
                                    if args.env == 'pdb':
                                        from distutils import spawn
                                        python_command = [spawn.find_executable("gdb"), '--args', PYCMD, '-m', 'pdb', tmp_filename]
                                        os.execv(python_command[0], python_command)
                                    py_output = subprocess.check_output(python_command, stderr=subprocess.STDOUT).decode('ascii')
                                py_success_number += 1
                                print('.'),
                            except subprocess.CalledProcessError as e:
                                py_output = ''
                                py_err = e.output.decode('ascii')
                                py_failure_number += 1
                                print('#Failure#'),
                            finally:
                                if py_output == '' and py_err == '':
                                    py_err = "An unknown error has occurred."

                        cpp_err = ''
                        clean_command = ''
                        if args.env in ['cpp', 'all',]:
                            # FIXME intermediate dir hardcoded to mpi
                            cpp_file_name = full_filename.replace('.py', '.cpp')
                            partial_path1, partial_path2 = cpp_file_name.rsplit('python', 1)
                            cpp_file_name = partial_path1 + 'cpp' + partial_path2
                            cpp_test_location = cpp_examples_location if run_examples else os.path.dirname(os.path.dirname(cpp_file_name))
                            results_dir = 'intel_intel64_parallel_' if run_examples else 'intel_intel64_'
                            results_dir_ext = 'dylib' if IS_MAC else 'so'
                            results_dir = results_dir + results_dir_ext

                            if IS_WIN:
                                exe_name = [jp(cpp_test_location, "x64", "Release.dynamic.threaded", filename_no_ext + ".exe")]
                                clean_command = ["rm", exe_name[0]] if os.path.isfile(exe_name[0]) else []
                                cd_cmd = ["cd", cpp_test_location, "&&"]
                                make_cmd = ["msbuild", "DAALExamples.sln", "/p:Configuration=Release.dynamic.threaded",
                                            "/p:Platform=x64", "/nr:false", "/m", "/t:Clean;" + filename_no_ext, "/nologo", "/verbosity:quiet"]
                                mpi_cmd = ['mpirun', '-n', '4', '-ppn', '1', '&&'] if cpp_test_location.endswith('mpi') else []
                                executable = [jp("x64", "Release.dynamic.threaded", filename_no_ext + ".exe")]
                                cpp_commands = cd_cmd + make_cmd + mpi_cmd + executable
                                if clean_command != '':
                                    subprocess.call(clean_command, shell=True)
                                make_subproc = subprocess.call(cd_cmd + make_cmd, shell=True)
                                if make_subproc:
                                    print("Make encountered an error:")
                                cpp_proc = subprocess.Popen(cd_cmd + mpi_cmd + exe_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                                cpp_output = cpp_proc.stdout.read().decode('ascii')
                                cpp_err = cpp_proc.stderr.read().decode('ascii')
                            else:
                                exe_name = jp(cpp_test_location, "_results", results_dir, filename_no_ext + '.exe')
                                clean_command = "rm " + exe_name if os.path.isfile(exe_name) else ""
                                cd_cmd = "cd " + cpp_test_location + " && "
                                name_option = 'example' if run_examples else 'sample'
                                if IS_LIN:
                                    lib_option = 'sointel64'
                                elif IS_MAC:
                                    lib_option = 'dylibintel64'

                                make_cmd = "make " + lib_option + " mode=build threading=parallel compiler=intel " + name_option + "=" + filename_no_ext
                                mpi_cmd = 'mpirun -n 4 -ppn 1 ' if cpp_test_location.endswith('mpi') else ''
                                executable = jp('_results', results_dir, filename_no_ext + '.exe')

                                if clean_command != '':
                                    subprocess.call([clean_command], shell=True)
                                make_subproc = subprocess.call([cd_cmd + make_cmd], shell=True)
                                if make_subproc:
                                    print("Make encountered an error:")
                                cpp_proc = subprocess.Popen([cd_cmd + mpi_cmd + exe_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                                cpp_output = cpp_proc.stdout.read().decode('ascii')
                                cpp_err = cpp_proc.stderr.read().decode('ascii')

                            if py_err == '':
                                print('.'),
                                cpp_success_number += 1
                            else:
                                cpp_failure_number += 1
                                print('#Failure#'),

                        if py_err != '' or cpp_err != '':
                            error_occurred = True
                            output_file.write("##########################################################\n" + filename_no_ext)
                            if args.env in ['py', 'all',]:
                                output_file.write(' '.join(python_command) + "\n")
                                output_file.write("Python Output:\n" + str(py_output) + str(py_err) + "\n\n")
                            if args.env in ['cpp', 'all',]:
                                output_file.write(' '.join(cd_cmd + make_cmd + mpi_cmd + executable) + "\n") if IS_WIN else output_file.write(cd_cmd + make_cmd + mpi_cmd + executable + "\n")
                                output_file.write("C++ Output:\n" + cpp_output + cpp_err + "\n\n")

                        # Pretty cool library that allows Python to compare two strings and show the differences between them. Then it writes
                        # the output of both .py and .exe and the differences in a text file under the same directory as the test scripts.
                        differences = ""
                        if not error_occurred and args.env == 'all':
                            py_cpp_test_passes += 1
                            from difflib import unified_diff
                            if filename_no_ext in ignoreoutput:
                                pyout  = list(filter(ignoreoutput[filename_no_ext], py_output.splitlines(False)))
                                cppout = list(filter(ignoreoutput[filename_no_ext], cpp_output.splitlines(False)))
                            else:
                                pyout  = list(filter(None, py_output.splitlines(False)))
                                cppout = list(filter(None, cpp_output.splitlines(False)))
                            diff = unified_diff(pyout, cppout)
                            diff_string = '\n'.join(diff)
                            if len(diff_string) > 0:
                                error_occurred = True
                                print('#Failure#'),
                                differences = "Differences:\n" + diff_string
                                output_file.write("##########################################################\n" + filename_no_ext)
                                output_file.write("Python Output:\n" + py_output + py_err + "\n\n")
                                output_file.write("C++ Output:\n" + cpp_output + cpp_err + "\n\n")
                                output_file.write(differences + "\n\n")
                            else:
                                differences_success_number += 1
                                print('.'),
                        print('')
                        if error_occurred:
                            print(strftime("%H:%M:%S", gmtime()) + '\tFAILED\t' + filename_no_ext)
                        else:
                            print(strftime("%H:%M:%S", gmtime()) + '\tPASSED\t' + filename_no_ext)
                        if args.verbose:
                            print("\n")
                            if args.env != 'cpp':
                                print(' '.join(python_command))
                                print("Python Output:\n" + str(py_output) + str(py_err) + "\n")
                            if args.env != 'py':
                                if IS_WIN:
                                    print(' '.join(cd_cmd + make_cmd + mpi_cmd + executable))
                                else:
                                    print(cd_cmd + make_cmd + mpi_cmd + executable)
                                print("C++ Output:\n" + cpp_output + cpp_err + "\n")
                            if not error_occurred and args.env == 'all':
                                print(differences + "\n")
                        error_occured = False
                        print("\n")

                        if tmp_filename != full_filename and os.path.exists(tmp_filename) and args.serialize_tests != 'keep':
                            os.remove(tmp_filename)

            if 'spark' in args.test:
                pyspark_dir = jp(py_samples_location, 'spark')
                os.chdir(pyspark_dir)
                delete_spark_results()

                # Run Pyspark samples
                subprocess.call(['./launcher.sh', 'intel64'])

                os.chdir(jp('..', '..', 'java', 'spark'))
                delete_spark_results()

                # Run Spark samples
                subprocess.call(['./launcher.sh', 'intel64'])

                # Verify results
                for spark_smpl in os.listdir('_results'):
                    spark_res = extract_spark_results(jp('_results', spark_smpl, spark_smpl + '.res'))
                    pyspark_res_path = jp(pyspark_dir, '_results', spark_smpl, spark_smpl + '.out')
                    pyspark_res = get_pyspark_results(pyspark_res_path)
                    compare_results(spark_smpl, pyspark_res, spark_res)
                    total_tests += 1
                    py_success_number += 1
                    java_success_number += 1

                os.chdir(toolscript_loc)

            if total_tests == 0:
                if '.py' in args.name:
                    print("Test name should not include '.py'")
                else:
                    print(args.name + " does not exist. Please use -s flag to show existing tests.")
                exit()
            print("##########################################################\nTests complete. Here are the following results:")
            output_file.write("##########################################################\nTests complete. Here are the following results:\n")
            if args.env != 'cpp':
                print("Python: " + str(py_success_number) + "/" + str(total_tests) + " succeeded." + (" Great!!!" if py_success_number == total_tests else ""))
                output_file.write("Python: " + str(py_success_number) + "/" + str(total_tests) + " succeeded." + (" Great!!!" if py_success_number == total_tests else "") + "\n")
            if args.env != 'py' and 'spark' not in args.test:
                print("C++:    " + str(cpp_success_number) + "/" + str(total_tests) + " succeeded." + (" Great!!!" if cpp_success_number == total_tests else ""))
                output_file.write("C++:    " + str(cpp_success_number) + "/" + str(total_tests) + " succeeded." + (" Great!!!" if cpp_success_number == total_tests else "") + "\n")
            if args.env == 'all' and py_cpp_test_passes != 0:
                print("Same output: " + str(differences_success_number) + "/" + str(py_cpp_test_passes) + " succeeded." + (" Great!!!" if differences_success_number == py_cpp_test_passes else ""))
                output_file.write("Same output: " + str(differences_success_number) + "/" + str(py_cpp_test_passes) + " succeeded." + (" Great!!!" if differences_success_number == py_cpp_test_passes else "") + "\n")
            print("\nAll output has been saved in " + test_output_file + "\n\n")
    # Writes any errors into a text file found under the SAT/daal/swig directory
    if len(error_template_string) > 0:
        error_template_string = "Couldn't digest the following occurences of what looks like template declarations:\n" + error_template_string
    with open(jp(os.path.abspath('.'), 'error.txt'), 'w') as error_file:
        error_file.write(str(error_collection_string))
        error_file.write(str(error_template_string))

    if args.doc_test:
        regex = r'<a name="([^\"]*)"></a>'
        grep_output = subprocess.check_output(r"grep -o -r -E '<a name=[^>]*></a>' ../daal/include", stderr=subprocess.STDOUT, shell=True).decode('ascii')
        lines = grep_output.split('\n')
        for line in lines:
            match = re.search(regex, line)
            if match:
                doc_string = match.group(1)
                try:
                    result = subprocess.check_output("grep -o -r " + doc_string + " doc/html/*.html", stderr=subprocess.STDOUT, shell=True).decode('ascii')
                except subprocess.CalledProcessError as err:
                    print("Doc string '{}' not found: #Failure#".format(doc_string))

    if len(error_collection_string) > 0 or len(error_template_string) > 0:
        print("----------------------------------------------------------------------------")
        print("The script has encountered unexpected situations. Here is the error log:")
        print("----------------------------------------------------------------------------")
        print(error_collection_string)
        print(error_template_string)
