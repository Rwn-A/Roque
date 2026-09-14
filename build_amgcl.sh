clang++ -std=c++17 -O2 -I./vendor -c c_wrappers/amgcl.cpp -o ./build/amgcl_wrapper.o
ar rcs ./build/amgcl.a ./build/amgcl_wrapper.o
