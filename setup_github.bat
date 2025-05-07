@echo off
echo Setting up Git repository for Trading Bot...

REM Initialize git repository if not already initialized
if not exist .git (
    echo Initializing Git repository...
    git init
    echo Git repository initialized successfully.
) else (
    echo Git repository already exists.
)

REM Add all files to git
echo Adding files to Git...
git add .

REM Initial commit
echo Creating initial commit...
git commit -m "Initial commit: Trading Bot with multiple strategies and Streamlit dashboard"

REM Instructions for connecting to GitHub
echo.
echo To push to GitHub:
echo.
echo 1. Create a new repository on GitHub without initializing it
echo 2. Run the following commands:
echo    git remote add origin https://github.com/07ItachiUchiha/trading-bot.git
echo    git branch -M main
echo    git push -u origin main
echo.

pause
