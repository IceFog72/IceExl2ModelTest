# IceExl2ModelTest
exl2 model tester

```
conda create -n IceExl2ModelTest python=3.11
conda activate IceExl2ModelTest
```

cd ...

```
conda install -c "nvidia/label/cuda-12.2.2" cuda
```

Run Setup.bat

Why using exllamav2-0.0.21? 

It's last version before mandatory flash attention in v0.1 ( yea you can disable it, but i couldn't make their mmlu eval to run, it's just crashing on my 2060 6gb )

I'm 0 in python, and this test only for comparing models. Test results are not intended to be compared with any other tests.



