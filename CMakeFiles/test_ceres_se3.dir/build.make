# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hcg/hcg/visual_navigation/visnav_new

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hcg/hcg/visual_navigation/visnav_new

# Include any dependencies generated for this target.
include CMakeFiles/test_ceres_se3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_ceres_se3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_ceres_se3.dir/flags.make

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o: CMakeFiles/test_ceres_se3.dir/flags.make
CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o: src/test_ceres_se3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hcg/hcg/visual_navigation/visnav_new/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o"
	/usr/bin/ccache /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o -c /home/hcg/hcg/visual_navigation/visnav_new/src/test_ceres_se3.cpp

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hcg/hcg/visual_navigation/visnav_new/src/test_ceres_se3.cpp > CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.i

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hcg/hcg/visual_navigation/visnav_new/src/test_ceres_se3.cpp -o CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.s

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.requires:

.PHONY : CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.requires

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.provides: CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_ceres_se3.dir/build.make CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.provides.build
.PHONY : CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.provides

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.provides.build: CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o


# Object files for target test_ceres_se3
test_ceres_se3_OBJECTS = \
"CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o"

# External object files for target test_ceres_se3
test_ceres_se3_EXTERNAL_OBJECTS =

test_ceres_se3: CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o
test_ceres_se3: CMakeFiles/test_ceres_se3.dir/build.make
test_ceres_se3: /home/hcg/hcg/visual_navigation/visnav/thirdparty/build-ceres-solver/lib/libceres.a
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libglog.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libspqr.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libtbb.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libcholmod.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libccolamd.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libcamd.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libcolamd.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libamd.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/liblapack.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libf77blas.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libatlas.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/librt.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libcxsparse.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/liblapack.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libf77blas.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libatlas.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/librt.so
test_ceres_se3: /usr/lib/x86_64-linux-gnu/libcxsparse.so
test_ceres_se3: CMakeFiles/test_ceres_se3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hcg/hcg/visual_navigation/visnav_new/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_ceres_se3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ceres_se3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_ceres_se3.dir/build: test_ceres_se3

.PHONY : CMakeFiles/test_ceres_se3.dir/build

CMakeFiles/test_ceres_se3.dir/requires: CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.requires

.PHONY : CMakeFiles/test_ceres_se3.dir/requires

CMakeFiles/test_ceres_se3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_ceres_se3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_ceres_se3.dir/clean

CMakeFiles/test_ceres_se3.dir/depend:
	cd /home/hcg/hcg/visual_navigation/visnav_new && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hcg/hcg/visual_navigation/visnav_new /home/hcg/hcg/visual_navigation/visnav_new /home/hcg/hcg/visual_navigation/visnav_new /home/hcg/hcg/visual_navigation/visnav_new /home/hcg/hcg/visual_navigation/visnav_new/CMakeFiles/test_ceres_se3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_ceres_se3.dir/depend

