#! /bin/bash
# Compile and build the executable 
# g++: Invokes the GNU C++ compiler.
# -c: Compiles and assembles the code into an object file (.o), skipping the linking phase.
# -g: Generates debugging information (for use with debuggers like gdb).
# -Wall: Enables all common compiler warning messages to help catch potential bugs.
# -o fem2d_heat_rectangle: Sets the name of the resulting output executable file

g++ -g -Wall fem2d_heat_rectangle.cpp  -o fem2d_heat_rectangle

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "Normal end of execution."

# Execute:

./fem2d_heat_rectangle


# rm rectangle_u00*.txt fem2d_heat_rectangle fem2d_*.eps

#------------------------------------------------------
# 4. Plot using python, the run create 10 png files, one for each u0000#.txt : 
#-----------------------------------------------------

python contour_sequence4.py rectangle_nodes.txt rectangle_u0000.txt


#rm rectangle_u00*.png, rectangle_u00*.txt fem2d*.o fem2d_heat_test

