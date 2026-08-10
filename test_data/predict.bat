@echo off
echo Sending prediction request...
curl -X POST http://192.168.0.40:8000/predict -H "Content-Type: application/json" -d @prediction_request.json
echo.
pause