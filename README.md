# Online Appendix of the paper "Between Predictive Quality and Energy: The Impact of In-Prompt Batch Size in LLM-Based Classification for Software Engineering Tasks"

In this paper, we assess the impact of the in-prompt batch size on the predictive quality and energy consumption of an LLM.
The appendix contains two main folders: experiment and raw_results.

## Folder: raw_results 
This folder consists of four subfolders: Task 1, Task 2, Data Selection, Cohen Kappa

### Subfolder Task 1
- `energy_info_run{i}.xlsx` presents raw energy consumed and carbon emission per model, dataset and batch size setting for run i, Task 1.
- `predictive_quality_scores_task1.xlsx` contains the predictive quality data of Task 1, including the adjusted F1 scores that take into account the missing predictions (F1_lower, F1_random, F2_upper).

#### Subsubfolder Post-hoc test
This folder contains two subfolders with the p-value results from the Nemenyi post-hoc test for Energy consumption and Predictive Quality.

### Subfolder Task 2
- `energy_info_run{i}.xlsx` presents raw energy consumed and carbon emission per model, dataset and batch size setting for run i, Task 2.
- `predictive_quality_scores_task2.xlsx` contains the predictive quality data of Task 2, including the adjusted F1 scores that take into account the missing predictions (F1_lower, F1_random, F2_upper).

#### Subsubfolder Post-hoc test
This folder contains two subfolders with the p-value results from the Nemenyi post-hoc test for Energy consumption and Predictive Quality.

### Subfolder: Dataset Selection
This subfolder contains the Excel file listing the papers considered for the experiments, along with the reasons for their exclusion.

### Subfolder: Cohen Kappa
This subfolder contains Cohen's Kappa scores for four datasets from Task 2.

### Subfolder: Related Work
This subfolder presents the venues considered for additional verification.

## Folder: experiment
This folder consists of four subfolders, which are used to perform the experiments with the four LLMs. Before running the experiment on your computer, make sure that the original datasets are in the root folder (as Excel files) [1][2], and your prompts in 'experiments/prompts' (including placeholders: m, n and project) as .txt file.
To perform the experiments for task 1, run the following command in your root folder:
```
python3.9 -m task_1.get_predictions_issues <output_folder_name>
```

To perform the experiments for task 2, run the following command in your root folder:
```
python3.9 -m task_2.get_predictions_reviews <output_folder_name>
```

### Subfolder: prompts
This subfolder is used to collect information related to prompts for each task. 
- `examples_task1.xlsx`: The demonstration examples for the classification task 1.
- `examples_task2.xlsx`: The demonstration examples for the classification task 2.
- `project_names.json`: For task 1, this file contains the correct formulation of each project name.
- `app_names.json`: For task 2, this file contains the correct formulation of each app name.
The prompts can be found in appendix A in the paper.

### Subfolder: task_1
This subfolder contains the main script `get_predictions_issues.py` for performing the experiment related to task 1.

### Subfolder: task_2
This subfolder contains the main script `get_predictions_reviews.py` for performing the experiment related to task 2.


### subfolder: utils
To execute the experiments in tasks 1 and 2, similar functions are used that are collected in the files:
- `generate_response.py`: helper modules for executing `get_predictions_issues.py` and `get_predictions_reviews.py`
- `gpu_energy_recorder.py`: helper script for measuring power, adapted from [3].


## Reference to data
- [1] Ashley T van Can and Fabiano Dalpiaz. 2025. Locating requirements in backlog items: Content analysis and experiments with large language models. Information and Software Technology 179 (2025), 107644.
- [2] Xiaodong Gu and Sunghun Kim. 2015. "What parts of your apps are loved by users?". In 2015 30th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 760–770.

## Other references
[3] Negar Alizadeh, Boris Belchev, Nishant Saurabh, Patricia Kelbert, and Fernando Castor. 2025. Language Models in Software Development Tasks: An Experimental Analysis of Energy and Accuracy. In 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR). IEEE, 725–736.
