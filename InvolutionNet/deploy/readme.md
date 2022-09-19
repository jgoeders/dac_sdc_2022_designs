## Compiling command for bbox computation C lib
## It may require the root account to compile this C code
g++ -shared -O3 yolo.cpp -o yolo.so -fPIC -lpthread -fpermissive

 
