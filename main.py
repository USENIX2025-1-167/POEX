import poex
from poex.POEX import POEX
import logging
from transformers import set_seed
from poex.datasets import JailbreakDataset
import argparse
import os
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
set_seed(42)

def main(args):

    # Load model and tokenizer
    model = poex.models.from_pretrained(
        args.attack_model_path,
        args.attack_model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=0.01
    )

    # Load dataset
    dataset = JailbreakDataset.load_csv(args.dataset_path)[:5]

    # API class initialization and settings
    attacker = POEX(
        attack_model=model,
        target_model=model,
        constraint_model = model,
        policy_evaluator = args.policy_evaluator,
        jailbreak_datasets=dataset,
        ppl_threshold=args.ppl_threshold,
        jailbreak_prompt_length=args.jailbreak_prompt_length,
        num_turb_sample=args.num_turb_sample,
        batchsize=args.batchsize,
        top_k=args.top_k,
        max_num_iter=args.max_num_iter,
        is_universal=args.is_universal
    )

    # Launch the attack
    attacker.attack()
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, f'poex_result_{args.attack_model_name}.jsonl')
    attacker.jailbreak_datasets.save_to_jsonl(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_model_name', type=str,  default='llama-3', help='Name of the attack model')
    parser.add_argument('--attack_model_path', type=str, default='models/Meta-Llama-3-8B-Instruct', help='Path to the attack model')
    parser.add_argument('--policy_evaluator', type=str, default='llama3',choices=['openai','llama3'], help='Policy evaluator')
    parser.add_argument('--dataset_path', type=str, default='datasets/harmful-rlbench.csv', help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default='results', help='Path to the output file')
    parser.add_argument('--ppl_threshold', type=int, default=200, help='Perplexity threshold')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens')
    parser.add_argument('--jailbreak_prompt_length', type=int, default=5, help='Length of the jailbreak prompt')
    parser.add_argument('--num_turb_sample', type=int, default=64, help='Number of samples for Turbo')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size')
    parser.add_argument('--top_k', type=int, default=256, help='Top k')
    parser.add_argument('--max_num_iter', type=int, default=1, help='Maximum number of iterations')
    parser.add_argument('--is_universal', type=bool, default=False, help='Is universal')

    args = parser.parse_args()

    main(args)
