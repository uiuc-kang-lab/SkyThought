import pandas as pd
from datasets import load_dataset

from base import TaskConfig, TaskHandler

# Fix circular import by importing from external legalbench package
# Use an absolute import path to avoid confusion with the local package
import legalbench.constants
from legalbench.utils import generate_prompts
from legalbench.evaluation import evaluate

# Access TASKS from the constants module
TASKS = legalbench.constants.TASKS
ISSUE_TASKS = legalbench.constants.ISSUE_TASKS


class LegalBenchTaskHandler(TaskHandler):
    """
    TaskHandler for LegalBench tasks using the forked repository
    at https://github.com/uiuc-kang-lab/legalbench
    """
    
    def __init__(self, task_config: TaskConfig):
        super().__init__(task_config)
        self.current_task = self.task_config.dataset_subset
        
    def generate_prompt(self, problem):
        """
        Generate a prompt for the given problem using LegalBench's template system
        """
        # Load the prompt template for the specific task
        from legalbench import tasks as lb_tasks_module
        import os
        
        # Get the base prompt template path
        legalbench_path = os.path.dirname(lb_tasks_module.__file__)
        template_path = os.path.join(os.path.dirname(legalbench_path), "tasks", self.current_task, "base_prompt.txt")
        
        with open(template_path, "r") as f:
            prompt_template = f.read()
        
        # Convert the single problem to a dataframe for compatibility with LegalBench
        problem_df = pd.DataFrame([problem])
        
        # Use LegalBench's prompt generation system
        prompts = generate_prompts(prompt_template, problem_df)
        return prompts[0]  # Return the first (and only) prompt

    def check_correctness(self, problem, generation):
        """
        Check the correctness of a generated response against the reference answer
        using LegalBench's evaluation method
        """
        # Use LegalBench's built-in evaluation function
        score = evaluate(
            self.current_task,
            [generation],  # LegalBench expects lists
            [problem[self.task_config.answer_key]]
        )
        
        # Handle different return types from evaluate()
        if isinstance(score, list):
            return score[0] >= 0.5  # Threshold for correctness
        else:
            # Single score (accuracy or F1)
            return score >= 0.5
            
    def update_results(self, problem, response):
        """
        Update the results dictionary with correctness evaluation
        """
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        
        curr_res = self.check_correctness(problem, generation=response)
        
        if curr_res:
            response_entry["correctness"] = True
            response_entry["reason"] = ""
        else:
            response_entry["correctness"] = False
            response_entry["reason"] = "Solution is incorrect."
            
        return response_entry

    def load_and_filter_dataset(self, start, end, split=None, subset=None, difficulty=None):
        """
        Load and filter the dataset based on the specified parameters
        """
        # If subset is provided, use it; otherwise use the one from config
        task_name = subset if subset else self.task_config.dataset_subset
        
        # Load dataset from HuggingFace
        dataset = load_dataset(
            "nguha/legalbench", 
            task_name,
            split=split if split else self.task_config.dataset_split
        ).to_pandas()
        
        # Return the specified slice
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
        
    @staticmethod
    def get_available_tasks():
        """
        Get a list of all available LegalBench tasks
        """
        return TASKS
    