@echo off
echo Sending prediction request...
curl -X POST http://localhost:8000/predict/cluster -H "Content-Type: application/json" -d @prediction_request.json
echo.
pause