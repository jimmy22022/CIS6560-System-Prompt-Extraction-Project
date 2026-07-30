# CIS6560 System Prompt Extraction Project

This is the repository of my final project for CIS6560 - Cybersecurity Project, supervised under Dr. Rozita Dara.  This project evaluates system prompt leakage rates under different extraction attacks.  It also evaluates the privacy risks of embedding sensitive information in system prompts.

# Features:
- Scripts to prepare the system prompt dataset.
- Script to generate attack cases.
- Script to run attack cases against proprietary and local models.
- Script to perform postprocessing on responses where required and run metrics on results files.
- There is also synthetic sensitive-information marker evaluation.
- Defensive scripts for instruction defense, sandwich defense, XML tagging, multi-turn dialogue defense, and system prompt filtering.

## System Requirements
- Windows
- CPython 3.13
- pip
- Git

Python 3.13 is recommended. Python 3.12 is the minimum supported version for
the currently pinned dependencies because `numpy==2.5.1` requires Python
3.12 or newer.

## Installation
- Open PowerShell.
- git clone https://github.com/jimmy22022/CIS6560-System-Prompt-Extraction-Project.git
- cd CIS6560-System-Prompt-Extraction-Project
- Open your IDE terminal and run the following commands:
- py -3.13 -m venv .venv
- .\.venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- python -m pip install -r requirements.txt

## Windows Long-Path Support
On Windows, enable Win32 long-path support before installing packages
if the virtual environment is stored inside a deeply nested project path.  This can be done by opening PowerShell as administrator and running:

New-ItemProperty `
  -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" `
  -Value 1 `
  -PropertyType DWORD `
  -Force

Restart Windows after changing this setting, then recreate the virtual
environment and run the installation again.

## Environment Variables

API keys are loaded from a local .env file.  This file is not included and must be manually created after creating your own API keys.  Once keys are obtained,
create a new file called .env in the project root and add the environment variables in the following structure:

OPENAI_API_KEY=  
ANTHROPIC_API_KEY=  
GEMINI_API_KEY=

## Script Execution Order

The scripts are numbered in the intended execution order:

Baseline attack pipeline:
1. step0_create_rag_prompts.py
2. step1_load_system_prompts.py
3. step2_load_attack_cases.py
4. step3_run_attacks_without_defenses.py
5. step9_load_metrics.py

Defense pipeline:
1. step4_run_instruction_defense.py
2. step5_run_sandwich_defense.py
3. step6_run_xml_tagging.py
4. step7_run_multi_turn_dialogue_defense.py
5. step8_run_system_prompt_filtering.py
6. step9_load_metrics.py

step9_load_metrics.py is run after step3_run_attacks_without_defenses.py to generate the appropriate processed baseline attack results
files required by step8_run_system_prompt_filtering.py.

Many of these experiments are paid API calls and take a substantial amount of time and money to run.  Review the selected
models, datasets, and number of attack cases before running them.