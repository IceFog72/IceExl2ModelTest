# IceExl2ModelTest

exl2 model tester (MMLU,Winogrande,MuSR,MMLU_PRO)

```
conda create -n IceExl2ModelTest python=3.11
conda activate IceExl2ModelTest
```

cd ... (to folder)

```
conda install -c "nvidia/label/cuda-12.2.2" cuda
```

Run Setup.bat

Why using exllamav2-0.0.21?

It's last version before mandatory flash attention in v0.1 ( yea we can disable it, but I couldn't make their new mmlu eval to run (it's just crashing on my 2060 6gb))

I'm 0 in python, and this test only for comparing models. Test results are not intended to be compared with any other tests.

Edit config.py and run main.py

exl2_eval.py - old all in 1 version.

If you editing code teleted to promt- don't forget delete already generated questions from cache_dir.

![image](https://github.com/IceFog72/IceExl2ModelTest/assets/164350516/d3f80163-17c7-4137-b5c7-0ff40c3effb2)


Feedback  [ST Discord Thread](https://discord.com/channels/1100685673633153084/1259572507157991474)
