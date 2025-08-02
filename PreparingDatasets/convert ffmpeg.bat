@echo off
set sourceFolder=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\1.RAW-AUDIO_TO_WAV\INPUT
set destinationFolder=C:\Users\wojtek\Desktop\AUDIO_PROCCESSING\1.RAW-AUDIO_TO_WAV\OUTPUT

if not exist "%destinationFolder%" mkdir "%destinationFolder%"

for %%f in ("%sourceFolder%\*.*") do (
    ffmpeg -i "%%f" -ar 22050 -ac 1 "%destinationFolder%\%%~nf.wav"
)

echo Przetwarzanie zakończone.
pause
