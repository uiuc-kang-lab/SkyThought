# Implementation Plan for Adding New Evaluation Datasets to SkyThought

This document outlines a comprehensive plan for adding six new evaluation datasets to the SkyThought framework:
1. LegalBench (https://hazyresearch.stanford.edu/legalbench/)
2. LexGLUE (https://huggingface.co/datasets/coastalcph/lex_glue)
3. MedQA (https://github.com/jind11/MedQA)
4. PubMedQA (https://pubmedqa.github.io/)
5. FinBen (https://github.com/The-FinAI/PIXIU)
6. TabFact (https://tabfact.github.io/)

## 1. Project Structure

Based on the existing SkyThought framework, we will implement these new evaluation datasets following the established patterns:

```
skythought/evals/tasks/
├── legalbench/               # New directory for LegalBench
│   ├── legalbench.yaml       # Task configuration
│   └── legalbench_handler.py # Task handler implementation
├── lexglue/                  # New directory for LexGLUE
│   ├── lexglue.yaml          # Task configuration
│   └── lexglue_handler.py    # Task handler implementation
├── medqa/                    # New directory for MedQA
│   ├── medqa.yaml            # Task configuration
│   └── medqa_handler.py      # Task handler implementation
├── pubmedqa/                 # New directory for PubMedQA
│   ├── pubmedqa.yaml         # Task configuration
│   └── pubmedqa_handler.py   # Task handler implementation
├── finben/                   # New directory for FinBen
│   ├── finben.yaml           # Task configuration
│   └── finben_handler.py     # Task handler implementation
└── tabfact/                  # New directory for TabFact
    ├── tabfact.yaml          # Task configuration
    └── tabfact_handler.py    # Task handler implementation
```

## 2. Implementation Steps

### 2.1 Common Implementation Steps for All Datasets

For each dataset, we will follow these common steps:

1. Create a new directory under `skythought/evals/tasks/`
2. Create YAML configuration file
3. Implement the task handler that extends `TaskHandler`
4. Add the task handler to `skythought/evals/tasks/__init__.py`
5. Test the implementation with a simple model

### 2.2 Dataset-Specific Implementation Details

#### 2.2.1 LegalBench

**Dataset Information:**
- Source: https://github.com/HazyResearch/legalbench
- HuggingFace dataset path: `nguha/legalbench`
- Available subsets: 162 legal reasoning tasks from 40 contributors
- Task types: Classification, extraction, multilabel classification, and free-form answers

**Implementation Notes:**
- Support for multiple legal task categories organized in task lists (`TASKS`, `ISSUE_TASKS`, etc.)
- Each task has train and test splits:
  - `train` split: Small set (usually <10 samples) for few-shot demonstrations
  - `test` split: Larger set for evaluation
- Handle task-specific prompt templates stored in text files with {{col_name}} placeholders
- Implement evaluation metrics - most tasks use balanced-accuracy, some use F1
- Support task filtering by license information

**YAML Configuration:**
```yaml
handler: legalbench
dataset_path: nguha/legalbench
dataset_subset: null  # Will be set dynamically based on the task (e.g., "abercrombie")
dataset_split: test
question_key: text
answer_key: answer
templating_parameters:
  template_path: "{task_dir}/base_prompt.txt"  # Path to task-specific prompt template
  use_few_shot: true  # Whether to use few-shot examples from train split
```

**Required Functions:**
```python
# Load tasks and organize by categories
def get_tasks():
    """Get all available LegalBench tasks."""
    from tasks import TASKS, ISSUE_TASKS  # Import task lists
    return TASKS

# Load task prompt template
def load_prompt_template(task_name):
    """Load the prompt template for a specific task."""
    with open(f"tasks/{task_name}/base_prompt.txt") as in_file:
        return in_file.read()

# Generate prompts using templates
def generate_prompts(prompt_template, data_df):
    """Generate prompts by applying the template to each row in the dataframe."""
    from utils import generate_prompts
    return generate_prompts(prompt_template, data_df)

# Evaluate model outputs
def evaluate_outputs(task_name, generations, references):
    """Evaluate model outputs against references for a specific task."""
    from evaluation import evaluate
    return evaluate(task_name, generations, references)

# Filter tasks by license
def filter_tasks_by_license(license_type="CC BY 4.0"):
    """Filter tasks based on their license type."""
    filtered_tasks = []
    for task in get_tasks():
        dataset = datasets.load_dataset("nguha/legalbench", task, split="train")
        if dataset.info.license == license_type:
            filtered_tasks.append(task)
    return filtered_tasks
```

#### 2.2.2 LexGLUE

**Dataset Information:**
- Source: https://github.com/coastalcph/lex-glue
- HuggingFace dataset path: `coastalcph/lex_glue`
- Available subsets: 7 datasets (ECtHR Task A & B, SCOTUS, EUR-LEX, LEDGAR, UNFAIR-ToS, CaseHOLD)
- Task type: Multi-class classification, multi-label classification, multiple-choice QA

**Implementation Notes:**
- Support for 7 distinct legal tasks covering ECHR, US Law, EU Law, and Contracts
- Implement specialized scoring for multi-label and multiple-choice tasks
- Handle different data formats for each subtask

**YAML Configuration:**
```yaml
handler: lexglue
dataset_path: coastalcph/lex_glue
dataset_subset: null  # Will be set dynamically based on the selected subtask
dataset_split: test
question_key: text
answer_key: label
templating_parameters:
  template: "Legal task: {task_description}\n\n{prompt}\n\nSelect the correct answer."
```

#### 2.2.3 MedQA

**Dataset Information:**
- Source: https://github.com/jind11/MedQA
- HuggingFace dataset path: `bigbio/medqa` or direct dataset
- Available subsets: USMLE (US Medical Licensing Examination), MCML (Mainland China), TWML (Taiwan)
- Task type: Multiple choice (4 options per question)

**Implementation Notes:**
- Handle multiple-choice medical questions with specialized formatting 
- Support for question format with average question length of 116.6 tokens for USMLE
- Implement appropriate answer extraction logic with letter-based answers (A,B,C,D)

**YAML Configuration:**
```yaml
handler: medqa
dataset_path: bigbio/medqa
dataset_subset: usmle  # Default to USMLE, can be changed
dataset_split: test
question_key: question
answer_key: answer_idx  # Uses letter index for the answer
templating_parameters:
  template: "Medical Question: {question}\n\nAnswer Choices:\nA: {options.A}\nB: {options.B}\nC: {options.C}\nD: {options.D}\n\nReturn your answer as a single letter."
```

#### 2.2.4 PubMedQA

**Dataset Information:**
- Source: https://pubmedqa.github.io/
- HuggingFace dataset path: `pubmed_qa`
- Available subsets: pqa_labeled (1K expert-labeled), pqa_unlabeled (61.2K), pqa_artificial (211.3K)
- Task type: Yes/No/Maybe binary classification with context

**Implementation Notes:**
- Handle research questions that require reasoning from biomedical abstracts
- Primary task involves answering yes/no/maybe questions based on provided context
- Implement context-aware evaluation based on the reasoning-required setting

**YAML Configuration:**
```yaml
handler: pubmedqa
dataset_path: pubmed_qa
dataset_subset: pqa_labeled
dataset_split: test
question_key: question
answer_key: final_decision
templating_parameters:
  template: "Context: {context}\n\nQuestion: {question}\n\nBased on the context, answer with 'yes', 'no', or 'maybe'."
```

#### 2.2.5 FinBen

**Dataset Information:**
- Source: https://github.com/The-FinAI/PIXIU
- HuggingFace dataset path: Various datasets under FinBen benchmark
- Available subsets: Multiple datasets across 7 categories, 36 datasets covering 24 tasks total
- Task type: Classification, regression, forecasting, and decision-making tasks

**Implementation Notes:**
- Support for 7 financial task categories: information extraction, textual analysis, QA, text generation, risk management, forecasting, and decision-making
- Implement diverse evaluation metrics: F1, Accuracy, MCC, ROUGE, BERTScore, Cumulative Return, Sharpe Ratio
- Handle specialized financial data formats and metrics

**YAML Configuration:**
```yaml
handler: finben
dataset_path: The-FinAI/finben
dataset_subset: null  # Will be set dynamically based on the task
dataset_split: test
question_key: input
answer_key: output
templating_parameters:
  template: "Financial Task: {task_description}\n\n{input}\n\nProvide your analysis or answer."
```

#### 2.2.6 TabFact

**Dataset Information:**
- Source: https://tabfact.github.io/
- HuggingFace dataset path: `tab_fact`
- Available subsets: Simple (r1) and Complex (r2) reasoning categories
- Task type: Table-based fact verification (ENTAILED/REFUTED binary classification)

**Implementation Notes:**
- Handle structured tabular data with 118,275 annotated statements over 16,573 Wikipedia tables
- Process table data into appropriate format for prompt construction
- Implement binary classification evaluation (ENTAILED=1/REFUTED=0)

**YAML Configuration:**
```yaml
handler: tabfact
dataset_path: tab_fact
dataset_subset: default
dataset_split: test
question_key: statement
answer_key: label
templating_parameters:
  template: "Table: {table}\n\nStatement: {statement}\n\nIs the statement 'ENTAILED' or 'REFUTED' based on the table data?"
```

## 3. Implementation Roadmap

### 3.1 Phase 1: Initial Setup (Week 1)
- Create directory structure for all six datasets
- Implement basic YAML configurations
- Set up task handlers with minimal functionality

### 3.2 Phase 2: Core Implementation (Week 2-3)
- Implement core functionality for all handlers
- Develop dataset loading and preprocessing logic
- Implement task-specific prompt templates
- Create evaluation metrics for each task

### 3.3 Phase 3: Testing & Refinement (Week 4)
- Test all implementations with small models
- Validate evaluation metrics against baseline results
- Refine implementations based on testing feedback
- Document any dataset-specific considerations

### 3.4 Phase 4: Integration & Documentation (Week 5)
- Integrate with the existing SkyThought framework
- Add comprehensive documentation for each dataset
- Create example scripts to demonstrate usage
- Run full evaluation suite with benchmark models

## 4. Potential Challenges and Mitigations

### 4.1 Dataset Consistency
**Challenge**: Some datasets may have inconsistent formats or special cases.
**Mitigation**: Implement robust data validation and preprocessing steps.

### 4.2 Complex Scoring Logic
**Challenge**: Specialized domains require domain-specific scoring criteria.
**Mitigation**: Consult domain expertise and implement flexible scoring systems.

### 4.3 Performance Considerations
**Challenge**: Some datasets might be large and cause performance issues.
**Mitigation**: Implement sampling strategies and optimize data loading.

## 5. Conclusion

This implementation plan provides a structured approach to integrate six new evaluation datasets into the SkyThought framework. By following this plan, we can ensure consistency, reliability, and comprehensive coverage of domain-specific evaluation benchmarks.
