@echo off
set root=%PY_HOME%
call %root%\Scripts\activate.bat  %root%\envs\tensorflow_gpu
python -m imagepy