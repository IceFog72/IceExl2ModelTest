# IceExl2ModelTest
![image](https://github.com/user-attachments/assets/9f1c71d5-2028-4889-9ae2-90be8f9f5f28)
exl2 model tester (using MMLU,Winogrande,HeLLaSWAG,MuSR,MMLU_PRO datasets)

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

I'm 0 in python, and this test only for comparing models. Test results are not intended to be compared with any other tests as implementation may be different.

Edit config.py and run main.py

exl2_eval.py - old all in 1 version.

If you editing code releted to promt- don't forget delete already generated questions from cache_dir.




Feedback  [Thread about it in SillyTavern Discord](https://discord.com/channels/1100685673633153084/1259572507157991474)
