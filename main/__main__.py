""" This script tries to bring together every dataset
So that we can have a single interface to access all the datasets
"""
import argparse

from main.mydatasets.commonsenseqa_dataset import CommonsenseQADataset
from .mydatasets import SapDataset, FantomDataset, TomiDataset, OpenTomDataset, AdvCsfbDataset, MindGamesDataset, SocialIQaDataset
from main.utils import get_total_cost

"""
Example usage
python3 -m main --dataset tomi --config ./config-azure.json --model gpt-3.5-azure --query-method cot --query-position end --kshots 0 --splitted-context 1 --num-experiments 300
"""

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", dest="config_file", type=str, default='./config-azure.json',
                    help="LLM config file.")
parser.add_argument("-m", "--model", dest="model_name", type=str, default='gpt-3.5-azure',
                    help="LLM we query. So far [gpt-3.5-azure, pythia-12B] are supported.")
parser.add_argument("-q", "--query-method", dest="query_method", type=str, default='cot',
                    help="Query method. So far [cot, cot-wm, tot, cot-wm-chat, struct, cot-explwm] are supported.")
parser.add_argument("-qp", "--query-position", dest="query_position", type=str, default='end',
                    help="Where the query appears. So far [beginning, end] are supported and only for tomi and fantom (for other datasets it is ignored).")
parser.add_argument("-k", "--kshots", dest="kshots", type=str, default='1',
                    help="Number of illustrations provided (k-shots). So far [0, 1] are supported.")
parser.add_argument("-n", "--num-experiments", dest="n_samples", type=int, default=10,
                    help="Number of examples to be queried.")
parser.add_argument("-r", "--random-sample", dest="random_sample", type=str, default="True",
                    help="Samples are taken randomly from the input distribution.")
parser.add_argument("-d", "--dataset", dest="dataset_name", type=str, default="tomi",
                    help="Dataset to be used. So far [tomi, sap, fantom, adv-csfb, mindgames] are supported.")
parser.add_argument("-sc", "--splitted-context", dest="num_splitted_context", type=int, default=1,
                    help="NOTE: this has effect only for cot-wm, for others it is ignored. Number of splitted context for cot-wm (otherwise it is ignored).")
parser.add_argument("-s", "--seed", dest="seed", type=int, default=-1,
                    help="Seed for reproducibility of the random sampling. Set negative for random seed.")
parser.add_argument("-i", "--input_type", dest="input_type", type=str, default="attitude", help="Input type")
parser.add_argument("-o", "--output_type", dest="output_type", type=str, default="multiple",
                    help="Answer required. So far [open, multiple] are supported.")
parser.add_argument("-b", "--batch", dest="batch", type=int, default=1,
                    help="Batch size. If set to value different than 1 tries to parallelize when possible (ask Angelo for when it is possible)")
parser.add_argument("-w", "--wandb", dest="has_wandb", default=False, action='store_true',
                    help="Whether to log the results to wandb. Default is False. If set to True, the config file should have the wandb key.")

# ToT variables
# parser.add_argument("--tot_steps", type=int, default=4, help="Number of steps in ToT")
parser.add_argument("--n_generate_sample", type=int, default=3, help="Number of samples to generate")
parser.add_argument("--method_generate", type=str, default='sample', help="Method to generate samples")
parser.add_argument("--method_evaluate", type=str, default='vote', help="Method to evaluate samples")
parser.add_argument("--method_select", type=str, default='greedy', help="Method to select samples")
parser.add_argument("--n_evaluate_sample", type=int, default=1, help="Number of samples to evaluate")
parser.add_argument("--n_select_sample", type=int, default=1, help="Number of samples to select")


args = parser.parse_args()

# Convert to dictionary
# args_dict = vars(args)
# print(args_dict)

match args.dataset_name:
    case "tomi":
        dataset = TomiDataset(args)
        prompts, questions, answers = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case "sap": # almost the same as Tomi.
        dataset = SapDataset(args)
        prompts, questions, answers = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case "fantom":
        dataset = FantomDataset(args)
        prompts, questions, answers = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case "adv-csfb":
        dataset = AdvCsfbDataset(args)
        prompts, questions, _, answers, types = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers, types)
    case "mindgames":
        dataset = MindGamesDataset(args, is_train=False)
        prompts, questions, answers = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case "socialiqa":
        dataset = SocialIQaDataset(args)
        prompts, questions, answers = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case "commonsenseqa":
        dataset = CommonsenseQADataset(args)
        prompts, questions, answers = dataset.get_dataset()
        print(answers[:5])
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case "opentom":
        dataset = OpenTomDataset(args)
        prompts, questions, answers = dataset.get_dataset()
        model_answers = dataset.run_experiments(prompts, questions, answers)
        dataset.save_results(model_answers, answers)
    case _:
        print("Invalid dataset name")

print("Total cost: ", get_total_cost())