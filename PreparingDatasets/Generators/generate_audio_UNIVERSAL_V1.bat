@echo off
setlocal enabledelayedexpansion

REM =============================
REM   Ustawienia ścieżek
REM =============================
set "CHECKPOINT_FILE=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\3.GAN_GENERATORS\HIFI_GAN\hifi-gan\pretrained\UNIVERSAL_V1\g_02500000"
set "INPUT_DIR=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\3.GAN_GENERATORS\HIFI_GAN\INPUT\UNIVERSAL_V1"
set "OUTPUT_DIR=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\3.GAN_GENERATORS\HIFI_GAN\OUTPUT\UNIVERSAL_V1"
set "SCRIPT_PATH=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\3.GAN_GENERATORS\HIFI_GAN\hifi-gan\inference_e2e.py"

REM =============================
REM   Tworzenie folderu wyjściowego, jeśli nie istnieje
REM =============================
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM =============================
REM   Uruchomienie inference_e2e.py dla całego katalogu INPUT_DIR
REM =============================
python "%SCRIPT_PATH%" --checkpoint_file="%CHECKPOINT_FILE%" ^
                       --input_mels_dir="%INPUT_DIR%" ^
                       --output_dir="%OUTPUT_DIR%"

if %errorlevel%==0 (
    echo Wszystkie pliki zostały przetworzone. Wyniki zapisano w folderze: %OUTPUT_DIR%
) else (
    echo Wystąpił błąd podczas przetwarzania. Sprawdź parametry i pliki wejściowe.
)

endlocal
pause
