@echo off

call conda deactivate
call conda activate IceExl2ModelTest

call cd /d %~dp0
call python setup.py %*
call pause