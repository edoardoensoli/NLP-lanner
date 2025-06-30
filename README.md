# Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools

Codes and Dataset for the Paper "[Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools (NAACL'25)](https://arxiv.org/pdf/2404.11891)".

## Ski Planning Extension

This repository has been extended with a **Ski Resort Planning** module that adapts the original travel planning framework for ski resort trip planning. The ski planning framework includes:

- **Dataset**: Comprehensive ski resort data with accommodation, slopes, equipment rental, and car rental information
- **APIs**: Specialized tools for ski resort search, slope information, equipment rental, and transportation
- **Planning**: Automated ski trip planning with budget constraints and preferences

The ski planning module is based on the same formal verification approach as the original travel planner but adapted for winter sports vacation planning. See `SKI_PLANNING_README.md` for detailed documentation.

**Key distinction**: In the ski planning context, "Resort" refers to ski facilities including ski pass costs, while "Accommodation" refers specifically to lodging arrangements.

## Framework
<img src="./imgs/fm_travelplanner_overview.png" width="800px" alt="s" />

## Setup Environment

1. Create a conda environment and install dependency:
    ```bash
    conda create -n fmtravelplanner python=3.9
    conda activate fmtravelplanner
    pip install -r requirements.txt
    ```

2. **Configure API Keys:**
    ```powershell
    # Copy the example configuration file
    cp config.example.py config.py
    
    # Edit config.py and add your API keys
    # Example:
    OPENAI_API_KEY = 'your_openai_api_key_here'
    CLAUDE_API_KEY = 'your_claude_api_key_here'  # if using Claude
    MIXTRAL_API_KEY = 'your_mixtral_api_key_here'  # if using Mixtral
    ```

3. The UnsatChristmas dataset is provided in `database_small` folder. You can run interactive plan repair experiment with this database.

4. To run satisfiable plan generation experiment, refer to paper "TravelPlanner: A Benchmark for Real-World Planning with Language Agents" and their github repo to download their database and train/validation/test set.

## Running
#### Satisfiable Plan Solving 
The file for satisfiable plan generation experiment is `test_travelplanner.py`. An example command is ```python test_travelplanner.py  --set_type train --model_name gpt```
*Note: You might want to use the training set to adjust the prompts for different LLMs. You can add customized checker for steps and codes to further improve the performance.*

#### Unsatisfiable Plan Repair 

* Run test_travelplanner_interactive.py for unsatisfiable interactive plan repair experiment for TravelPlanner. Follow the instructions in file to first collect initial codes and then do the plan repair. 
* Run test_unsat.py for unsatisfiable interactive plan repair experiment for UnsatChristmas. Follow the instructions in file to first collect initial codes and then do the plan repair. 

## Evaluation
For satisfiable plan solving, after running experiments for a dataset (train/ validation/ test), run `convert_json.py` to convert the generated plan txt file to json files, and then run `collect_plans.py` to collect all plans to form a single jsonl file. Then, you can use this jsonl file and follow TravelPlanner's evaluation method to evaluate (either through [leaderboard](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard) or through `eval.py` in TravelPlanner codebase)

For unsatisfiable plan repair, you can use the `check_plans()` function in `collect_plans.py` to check for successful plan generation.

## Prompts
The prompts we used are included in the `prompts` folder

## Citation
```md
@article{hao2024large,
  title={Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools},
  author={Hao, Yilun and Chen, Yongchao and Zhang, Yang and Fan, Chuchu},
  journal={arXiv preprint arXiv:2404.11891},
  year={2024}
}
```