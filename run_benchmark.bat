@echo off
REM Run tax benchmarking with Gemini model
python tax_ai_benchmarking_new.py ^
  --gt "%~1" ^
  --candidate "Gemini=%~2" ^
  --out "evaluation_results_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.xlsx" ^
  --max-workers 10 ^
  --llm "gemini-2.5-flash-preview-05-20"

