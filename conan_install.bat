conan install . -of build -u -b missing -o shared=True
cmake -Dfmus=WindToPower;WindGenerator --preset conan-default 
cd build && cmake --build . -j 14 --config Release 