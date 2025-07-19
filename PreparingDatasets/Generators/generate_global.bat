@echo off

REM =============================
REM   Ustawienia ścieżek
REM =============================
set "GENERATORS_DIR=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\3.GAN_GENERATORS\HIFI_GAN\Generators"

REM =============================
REM   Uruchamianie generatorów
REM =============================
echo Uruchamianie wszystkich generatorów...

start /B "" "%GENERATORS_DIR%\generate_audio_LJ_V1.bat"
start /B "" "%GENERATORS_DIR%\generate_audio_LJ_V2.bat"
start /B "" "%GENERATORS_DIR%\generate_audio_UNIVERSAL_V1.bat"
start /B "" "%GENERATORS_DIR%\generate_audio_VCTK_V1.bat"
start /B "" "%GENERATORS_DIR%\generate_audio_VCTK_V2.bat"

echo Wszystkie generatory zostały uruchomione.
pause