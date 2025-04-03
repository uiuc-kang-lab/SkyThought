import argparse
import sys

from skythought.evals.tasks.legalbench_eval.legalbench_handler import LegalBenchTaskHandler
from skythought.evals.tasks.base import TaskConfig
from legalbench import TASKS
from legalbench.evaluation import MANUAL_EVAL_TASKS

def run_task(task_name: str, num_examples: int = 5):
    """
    Run a single LegalBench task.
    
    Args:
        task_name: Name of the LegalBench task to use
        num_examples: Number of examples to load
    """
    # Create a task config
    task_config = TaskConfig(
        handler="legalbench",
        dataset_path="nguha/legalbench",
        dataset_subset=task_name,
        dataset_split="test",
        question_key="text",
        answer_key="answer",
    )
    
    # Create a task handler
    task_handler = LegalBenchTaskHandler(task_config)
    
    # Load some examples from the dataset
    data = task_handler.load_and_filter_dataset(0, num_examples)
    
    print(f"\nLoaded {len(data)} examples from the '{task_name}' task")
    
    # Display information about the examples
    for i, example in enumerate(data.to_dict("records")):
        print(f"\nExample {i+1}:")
        
        # Generate the prompt
        prompt = task_handler.generate_prompt(example)
        print("-" * 50)
        print("Prompt:")
        print(prompt)
        print("-" * 50)
        
        # Show the expected answer
        print(f"Expected answer: {example.get(task_config.answer_key, 'N/A')}")
        print("-" * 50)
        
        # Simulate a model response for demonstration
        # In a real scenario, you would call your LLM here
        simulated_response = example.get(task_config.answer_key, "Unknown")
        
        # Check correctness
        is_correct = task_handler.check_correctness(example, simulated_response)
        print(f"Simulated response: {simulated_response}")
        print(f"Is correct: {is_correct}")
        print("=" * 80)

def main(task_name: str, num_examples: int = 5):
    """
    Run LegalBench task(s).
    
    Args:
        task_name: Name of the LegalBench task to use or 'all' to run all tasks
        num_examples: Number of examples to load
    """

    if task_name.lower() == "all":
        # Run all tasks except excluded ones
        for task in TASKS:
            if task not in MANUAL_EVAL_TASKS:
                print(f"\n\n{'#' * 80}")
                print(f"# Running task: {task}")
                print(f"{'#' * 80}\n")
                try:
                    run_task(task, num_examples)
                except Exception as e:
                    print(f"Error running task {task}: {e}")
    else:
        # Run a single task
        run_task(task_name, num_examples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LegalBench Example")
    parser.add_argument(
        "--task", 
        type=str, 
        default="all",
        help="LegalBench task name to use or 'all' to run all tasks"
    )
    parser.add_argument(
        "--examples", 
        type=int, 
        default=3,
        help="Number of examples to load"
    )
    parser.add_argument(
        "--list-tasks", 
        action="store_true",
        help="List available tasks and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_tasks:
        print(f"Available LegalBench tasks ({len(TASKS)}):")
        # Print in columns
        for i in range(0, len(TASKS), 4):
            row = TASKS[i:i+4]
            print("  ".join(f"{task:<20}" for task in row))
        sys.exit(0)
        
    main(args.task, args.examples) 