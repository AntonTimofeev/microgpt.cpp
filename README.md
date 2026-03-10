# MICROGPT.CPP

A C++ implementation of a minimal GPT model inspired by Andrej Karpathy’s [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), using only the C++ standard library and a simple memory arena allocator.

The focus is on readability (and optimization) rather than minimizing line count.

# Build & Run

```
g++ -std=c++20 -O3 -ffast-math -march=native -mtune=native microgpt.cpp -o microgpt
time ./microgpt
```

You can add `-DDEBUG` to see how it learns.

# Performance

Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) was never meant to be fast, its goal is extreme readability and making it easy to learn how a GPT works from scratch. This C++ version focuses on optimization while remaining reasonably readable (barely at this point), because optimization is fun.

All tests run on `Intel Core i9-12900K`

### 16x16 network, 10000 steps
| Implementation | Time | vs PyPy JIT |Compilation flags|
|---|---|---|---|
|Python3|10m31,441s (631.441s)|0.23x||
|PyPy3|2m24,174s (144.174s)|1x||
|[mplekh/rust-microgpt](https://github.com/mplekh/rust-microgpt/tree/94005e239d99382046190dd01d60a85e7b17c13b)|0m1,070s (1.07s)|134x|`RUSTFLAGS="-C target-cpu=native" cargo run --release`|
|[Charbel199/microgpt.cpp](https://github.com/Charbel199/microgpt.cpp/tree/fc455d04fd89f49d22e183b86a51d0be3ba0e501)|0m0,893s (0.893s)|161x|`g++ -std=c++17 -Ofast -march=native -mtune=native microgpt.cpp -o microgpt`|
|[AntonTimofeev/microgpt.cpp](https://github.com/AntonTimofeev/microgpt.cpp/tree/e592f1d5f674f3da20c95e3bd99065bd46be74f6)|0m0,813s (0.813s)|177x|`g++ -std=c++20 -O3 -ffast-math -march=native -mtune=native microgpt.cpp -o microgpt`|
|[mplekh/rust-matrixmicrogpt](https://github.com/mplekh/rust-matrixmicrogpt/tree/dab1fef8908235a8a4d5b73e962b2fe61e89af25)|0m0,707s (0.707s)|204x|`RUSTFLAGS="-C target-cpu=native" cargo run --release`|
|[vixhal-baraiya/microgpt-c](https://github.com/vixhal-baraiya/microgpt-c/tree/43b3b24c781d65057f7b0e1296affd3a68a41b15)|0m0,027s (0.027s)|5339x|`gcc -O3 -ffast-math -march=native -mtune=native -o microgpt microgpt.c -lm`|


`microgpt-c` uses pre-allocated arrays for intermediate results, and different approach in computing gradients. Because of that approach it is quite difficult to acheive same output as in python, but I'll try to do that.

C/C++ results are a 10 runs average time.

```
$ python3 --version
Python 3.10.12

$ pypy3 --version
Python 3.8.13 (7.3.9+dfsg-1ubuntu0.1, Nov 15 2022, 06:22:50)
[PyPy 7.3.9 with GCC 11.3.0]

$ rustc --version
rustc 1.93.1 (01f6ddf75 2026-02-11)

$ g++ --version
g++ (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0
Copyright (C) 2022 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```