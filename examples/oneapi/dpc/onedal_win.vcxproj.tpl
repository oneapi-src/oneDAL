<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug.static.sequential|x64">
      <Configuration>Debug.static.sequential</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug.static.threaded|x64">
      <Configuration>Debug.static.threaded</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug.dynamic.sequential|x64">
      <Configuration>Debug.dynamic.sequential</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug.dynamic.threaded|x64">
      <Configuration>Debug.dynamic.threaded</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release.static.sequential|x64">
      <Configuration>Release.static.sequential</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release.static.threaded|x64">
      <Configuration>Release.static.threaded</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release.dynamic.sequential|x64">
      <Configuration>Release.dynamic.sequential</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release.dynamic.threaded|x64">
      <Configuration>Release.dynamic.threaded</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{example_guid}</ProjectGuid>
    <RootNamespace>{example_name}</RootNamespace>
    <ProjectName>{example_name}</ProjectName>
    <WindowsTargetPlatformVersion>10.0.17134.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.sequential|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <WholeProgramOptimization>false</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.threaded|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <WholeProgramOptimization>false</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.sequential|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <WholeProgramOptimization>false</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.threaded|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <WholeProgramOptimization>false</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.sequential|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>frue</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.threaded|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.sequential|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>frue</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.threaded|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>frue</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>Intel(R) oneAPI DPC++ Compiler</PlatformToolset>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings">
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.threaded|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.sequential|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.threaded|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.sequential|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.threaded|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.sequential|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.threaded|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.sequential|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.threaded|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.sequential|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.threaded|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.sequential|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.threaded|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.sequential|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.threaded|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.sequential|x64'">
    <LibraryPath>$(SolutionDir)..\..\..\lib\intel64;$(SolutionDir)..\..\..\..\..\tbb\latest\lib\intel64\vc_mt;$(LibraryPath)</LibraryPath>
    <ExecutablePath>$(ExecutablePath)</ExecutablePath>
    <IntDir>$(Platform)\$(Configuration)\</IntDir>
    <OutDir>$(SolutionDir)$(Platform)\$(Configuration)\</OutDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.threaded|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>None</DebugInformationFormat>
      <SYCLOptimization>MaxSpeed</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd_dll.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release.dynamic.sequential|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>None</DebugInformationFormat>
      <SYCLOptimization>MaxSpeed</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd_dll.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.threaded|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>None</DebugInformationFormat>
      <SYCLOptimization>MaxSpeed</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd.lib;onedal_cored.lib;onedal_threadd.lib;OpenCL.lib;onedal_sycld.lib;tbb12_debug.lib;tbbmalloc_debug.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release.static.sequential|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>None</DebugInformationFormat>
      <SYCLOptimization>MaxSpeed</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd.lib;onedal_cored.lib;onedal_sequentiald.lib;OpenCL.lib;onedal_sycld.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.threaded|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <SYCLOptimization>Disabled</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd_dll.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug.dynamic.sequential|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <SYCLOptimization>Disabled</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd_dll.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.sequential|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <SYCLOptimization>Disabled</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd.lib;onedal_cored.lib;onedal_sequentiald.lib;OpenCL.lib;onedal_sycld.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug.static.threaded|x64'">
    <ClCompile>
      <AdditionalIncludeDirectories>$(SolutionDir)..\..\..\include;$(SolutionDir)source</AdditionalIncludeDirectories>
      <EnableSyclOffload>true</EnableSyclOffload>
      <SYCLWarningLevel>DisableAllWarnings</SYCLWarningLevel>
      <WarningLevel>Level3</WarningLevel>
      <PreprocessorDefinitions>_MBCS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <SYCLOptimization>Disabled</SYCLOptimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <AdditionalOptions>/MDd</AdditionalOptions>
    </ClCompile>
    <Link>
      <TreatWarningAsError />
      <SYCLShowVerboseInformation>false</SYCLShowVerboseInformation>
      <AdditionalDependencies>onedal_dpcd.lib;onedal_cored.lib;onedal_threadd.lib;OpenCL.lib;onedal_sycld.lib;tbb12_debug.lib;tbbmalloc_debug.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalOptions>/link /ignore:4078 %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="$(ProjectDir)..\..\{example_relative_path}" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets">
  </ImportGroup>
</Project>
