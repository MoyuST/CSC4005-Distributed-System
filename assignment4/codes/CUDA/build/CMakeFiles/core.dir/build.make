# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /pvfsmnt/118010224/hw4/CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /pvfsmnt/118010224/hw4/CUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/core.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/core.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/core.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/core.dir/flags.make

CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o: ../imgui/backends/imgui_impl_opengl2.cpp
CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o -MF CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o.d -o CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/backends/imgui_impl_opengl2.cpp

CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/backends/imgui_impl_opengl2.cpp > CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.i

CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/backends/imgui_impl_opengl2.cpp -o CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.s

CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o: ../imgui/backends/imgui_impl_sdl.cpp
CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o -MF CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o.d -o CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/backends/imgui_impl_sdl.cpp

CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/backends/imgui_impl_sdl.cpp > CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.i

CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/backends/imgui_impl_sdl.cpp -o CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.s

CMakeFiles/core.dir/imgui/imgui.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/imgui.cpp.o: ../imgui/imgui.cpp
CMakeFiles/core.dir/imgui/imgui.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/core.dir/imgui/imgui.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/imgui.cpp.o -MF CMakeFiles/core.dir/imgui/imgui.cpp.o.d -o CMakeFiles/core.dir/imgui/imgui.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/imgui.cpp

CMakeFiles/core.dir/imgui/imgui.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/imgui.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/imgui.cpp > CMakeFiles/core.dir/imgui/imgui.cpp.i

CMakeFiles/core.dir/imgui/imgui.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/imgui.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/imgui.cpp -o CMakeFiles/core.dir/imgui/imgui.cpp.s

CMakeFiles/core.dir/imgui/imgui_demo.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/imgui_demo.cpp.o: ../imgui/imgui_demo.cpp
CMakeFiles/core.dir/imgui/imgui_demo.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/core.dir/imgui/imgui_demo.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/imgui_demo.cpp.o -MF CMakeFiles/core.dir/imgui/imgui_demo.cpp.o.d -o CMakeFiles/core.dir/imgui/imgui_demo.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_demo.cpp

CMakeFiles/core.dir/imgui/imgui_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/imgui_demo.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_demo.cpp > CMakeFiles/core.dir/imgui/imgui_demo.cpp.i

CMakeFiles/core.dir/imgui/imgui_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/imgui_demo.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_demo.cpp -o CMakeFiles/core.dir/imgui/imgui_demo.cpp.s

CMakeFiles/core.dir/imgui/imgui_draw.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/imgui_draw.cpp.o: ../imgui/imgui_draw.cpp
CMakeFiles/core.dir/imgui/imgui_draw.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/core.dir/imgui/imgui_draw.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/imgui_draw.cpp.o -MF CMakeFiles/core.dir/imgui/imgui_draw.cpp.o.d -o CMakeFiles/core.dir/imgui/imgui_draw.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_draw.cpp

CMakeFiles/core.dir/imgui/imgui_draw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/imgui_draw.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_draw.cpp > CMakeFiles/core.dir/imgui/imgui_draw.cpp.i

CMakeFiles/core.dir/imgui/imgui_draw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/imgui_draw.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_draw.cpp -o CMakeFiles/core.dir/imgui/imgui_draw.cpp.s

CMakeFiles/core.dir/imgui/imgui_tables.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/imgui_tables.cpp.o: ../imgui/imgui_tables.cpp
CMakeFiles/core.dir/imgui/imgui_tables.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/core.dir/imgui/imgui_tables.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/imgui_tables.cpp.o -MF CMakeFiles/core.dir/imgui/imgui_tables.cpp.o.d -o CMakeFiles/core.dir/imgui/imgui_tables.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_tables.cpp

CMakeFiles/core.dir/imgui/imgui_tables.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/imgui_tables.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_tables.cpp > CMakeFiles/core.dir/imgui/imgui_tables.cpp.i

CMakeFiles/core.dir/imgui/imgui_tables.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/imgui_tables.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_tables.cpp -o CMakeFiles/core.dir/imgui/imgui_tables.cpp.s

CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o: ../imgui/imgui_widgets.cpp
CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o -MF CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o.d -o CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_widgets.cpp

CMakeFiles/core.dir/imgui/imgui_widgets.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/imgui_widgets.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_widgets.cpp > CMakeFiles/core.dir/imgui/imgui_widgets.cpp.i

CMakeFiles/core.dir/imgui/imgui_widgets.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/imgui_widgets.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/imgui_widgets.cpp -o CMakeFiles/core.dir/imgui/imgui_widgets.cpp.s

CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o: ../imgui/misc/cpp/imgui_stdlib.cpp
CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o -MF CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o.d -o CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/misc/cpp/imgui_stdlib.cpp

CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/misc/cpp/imgui_stdlib.cpp > CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.i

CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/misc/cpp/imgui_stdlib.cpp -o CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.s

CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o: ../imgui/misc/freetype/imgui_freetype.cpp
CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o -MF CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o.d -o CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o -c /pvfsmnt/118010224/hw4/CUDA/imgui/misc/freetype/imgui_freetype.cpp

CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.i"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pvfsmnt/118010224/hw4/CUDA/imgui/misc/freetype/imgui_freetype.cpp > CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.i

CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.s"
	/opt/rh/devtoolset-10/root/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pvfsmnt/118010224/hw4/CUDA/imgui/misc/freetype/imgui_freetype.cpp -o CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.s

# Object files for target core
core_OBJECTS = \
"CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o" \
"CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o" \
"CMakeFiles/core.dir/imgui/imgui.cpp.o" \
"CMakeFiles/core.dir/imgui/imgui_demo.cpp.o" \
"CMakeFiles/core.dir/imgui/imgui_draw.cpp.o" \
"CMakeFiles/core.dir/imgui/imgui_tables.cpp.o" \
"CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o" \
"CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o" \
"CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o"

# External object files for target core
core_EXTERNAL_OBJECTS =

libcore.a: CMakeFiles/core.dir/imgui/backends/imgui_impl_opengl2.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/backends/imgui_impl_sdl.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/imgui.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/imgui_demo.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/imgui_draw.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/imgui_tables.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/imgui_widgets.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/misc/cpp/imgui_stdlib.cpp.o
libcore.a: CMakeFiles/core.dir/imgui/misc/freetype/imgui_freetype.cpp.o
libcore.a: CMakeFiles/core.dir/build.make
libcore.a: CMakeFiles/core.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library libcore.a"
	$(CMAKE_COMMAND) -P CMakeFiles/core.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/core.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/core.dir/build: libcore.a
.PHONY : CMakeFiles/core.dir/build

CMakeFiles/core.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/core.dir/cmake_clean.cmake
.PHONY : CMakeFiles/core.dir/clean

CMakeFiles/core.dir/depend:
	cd /pvfsmnt/118010224/hw4/CUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /pvfsmnt/118010224/hw4/CUDA /pvfsmnt/118010224/hw4/CUDA /pvfsmnt/118010224/hw4/CUDA/build /pvfsmnt/118010224/hw4/CUDA/build /pvfsmnt/118010224/hw4/CUDA/build/CMakeFiles/core.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/core.dir/depend

