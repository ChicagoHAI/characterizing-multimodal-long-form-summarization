import sys
sys.path.append('../')

import json
from utils.lm_api import LanguageModelAPI
import os
import tqdm
import re
import random
import argparse
import numpy as np

from transformers import PreTrainedTokenizerFast
import tiktoken


SIMPLE_PROMPT = "Summarize the following report.\n" + "MD&A: "

CLAUDE_PROMPT0 = "The following is an MD&A report:\n\n"
CLAUDE_PROMPT1 = "\n\nPlease summarize this report."

PROMPT_NUM = "Please include specific numeric values and key statistics."   # 2
PROMPT_TAB = "Please include the numeric values in the tables."             # 3
PROMPT_CoT_zshot = '''
Let's generate the summary step by step.

1. Read through the entire MD&A report carefully to understand the context.
2. Identify and extract the key topics and insights discussed in the report.
3. Pay attention to any tables presenting numeric data, such as income statements, balance sheets, or cash flow statements.
4. When including numbers in the summary, ensure they are: 
    a) Explicitly stated values from the original report (do not fabricate numbers).
    b) Stemmed from step-by-step verified calculations.
    c) Correctly rounded.
    d) Appropriately represented with clear context from the original source.
5. Synthesize the extracted information and numbers into a concise summary that flows logically.

Summary:
'''

encoding_openai = tiktoken.get_encoding('cl100k_base')
encoding_claude = PreTrainedTokenizerFast(tokenizer_file='../dataset/claude-v1-tokenization.json')
encoding = {
    'openai': encoding_openai,
    'claude': encoding_claude
}

length_dict = {
    'gpt-3.5-turbo-1106': 16385,    # tl
    'gpt-4-1106-preview': 128000,   # tl
    'claude-2.0': 100000,           # tl
    'claude-2.1': 200000,           # tl
    'command': 10000                # wl
    }

prompt_dict = {
    2: PROMPT_NUM,
    3: PROMPT_TAB,
    4: PROMPT_NUM + PROMPT_TAB,
    5: PROMPT_CoT_zshot
    }


def parse_args():

    parser = argparse.ArgumentParser(description='args for generating summary')
    parser.add_argument('--api', type=str, default='claude', 
                        choices=['claude', 'openai', 'cohere'], help='api type')
    parser.add_argument('--model', type=str, default='claude-2.1', 
                        choices=['gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'command', 'claude-2.0', 'claude-2.1'], help='model type')
    parser.add_argument('--shuffled', action='store_true', 
                        help='shuffle the report')
    parser.add_argument('--prompt', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5], help='prompt version')
    parser.add_argument('--report_num', type=int, default=1000, 
                        help='number of reports to do summarization')
    parser.add_argument('--output_base', type=str, 
                        help='output base for summarization result')
    args = parser.parse_args()

    args.output_base = f'../dataset/SEC-10k/LLM_SUMMERIZATION_ITEM7{"_SHUFFLED" if args.shuffled else ""}{"_P" if args.prompt!=0 else ""}{args.prompt if args.prompt!=0 else ""}'
    print(args.output_base)

    return args


def summarize_10k_varlen(args,
                         lm_api: LanguageModelAPI, 
                         file_path: str, 
                         file_notable_path: str, 
                         output_dir: str):

    output_filepath = f"{output_dir}/{file_path.split('/')[-1]}"

    if os.path.exists(output_filepath):
        print(f"File {file_path} already exists. Skipping ...")
        return 0
    
    res_dict = {}
    chunked_10k = json.load(open(file_path, "r")) # Load whole report
    if chunked_10k["item_7"] == "" or len(chunked_10k['item_7']) < 1700: # Check for invalid item_7
        print(f"Invalid item_7 for report {file_path} . Skipping ...")
        return 0
    
    report = chunked_10k["item_7"]
    res_dict['htm_filing_link'] = chunked_10k['htm_filing_link'] # report link
    res_dict['item_7'] = report # original item_7

    truncation_flag = 0
    # Truncate report to certain token length
    tokens = encoding[args.api].encode(report)[:length_dict[args.model]-500]
    report_truncated = encoding[args.api].decode(tokens)
    if len(tokens) == length_dict[args.model]-500:
        print('Truncated report')
        truncation_flag = 1

    # Get true tables and text
    report_truncated, text, tables = get_true_tbl_txt(file_notable_path, report_truncated)

    # Get shuffled report
    if args.shuffled:
        report_truncated, text, tables = get_shuffled_report(file_path, text, tables)
    
    res_dict['item_7_truncated'] = report_truncated
    res_dict['item_7_text'] = text
    res_dict['item_7_tables'] = ''.join(tables)

    # Customize chat_input for different kinds of api
    if args.prompt == 0:
        if args.api == 'cohere':
            chat_input = report_truncated
        elif args.api == 'openai':
            chat_input = SIMPLE_PROMPT + report_truncated
        elif args.api == 'claude':
            chat_input = CLAUDE_PROMPT0 + report_truncated + CLAUDE_PROMPT1
    else:
        chat_input = SIMPLE_PROMPT + report_truncated + '\n\n' + prompt_dict[args.prompt]
    
    # Use api to summarize
    res = lm_api.chat(chat_input)
    res_dict['summary'] = res

    json.dump(res_dict, open(output_filepath, "w"))
    return truncation_flag


def get_true_tbl_txt(file_notable_path, report):
    
    # Truncated text
    text = report
    
    # Original text
    report_notable = json.load(open(file_notable_path, 'r'))
    ori_text = report_notable['item_7']

    # Tables
    table_pattern = re.compile(r'Table \d+: <table>.*?</table>', re.DOTALL)
    tables = table_pattern.findall(report)
    # The last table is truncated
    tbl_tag0 = re.findall('<table>', report)
    tbl_tag1 = re.findall('</table>', report)
    if len(tbl_tag0) - len(tbl_tag1) == 1:
        matches = list(re.finditer(r'Table \d+: <table>', report))
        if matches:
            last_match = matches[-1]
            idx = last_match.start()
            if report[idx:].find('</table>') == -1:
                tables.append(report[idx:])
    
    # Get clean tables without table tags
    clean_tables = [re.sub(r'<table>|</table>|<tr>|<td>|</tr>|</td>|Table \d+: ', ' ', t) for t in tables]
    clean_tables = [re.sub(r'\s+', ' ', t) for t in clean_tables]
    true_tables = []
    for i in range(len(clean_tables)):
        # false table: replace the false table in the report and text to clean table without table tags
        if ori_text.find(clean_tables[i][6:-6]) != -1 or ori_text.find(clean_tables[i][-80:-6]) != -1:
            report = report.replace(tables[i], clean_tables[i])
            text = text.replace(tables[i], clean_tables[i])
            continue
        # clean table cannot be found in the original report: true table or wrongly recognized table
        else:
            # true tables
            if text.find(tables[i]) != -1:
                true_tables.append(tables[i])
                text = text.replace(tables[i], '')
            # wrong recoginized tables
            else:
                print('$'*10)
                print(repr(tables[i]))
    
    return report, text, true_tables


def get_shuffled_report(file_path, text, true_tables):

    sd = int(file_path.split('/')[-1].split('_')[0]+file_path.split('/')[-1].split('_')[2])
        
    shuffled_text = text.split('\n')
    random.seed(sd)
    random.shuffle(shuffled_text)
    
    shuffled_tables = true_tables.copy()
    random.seed(sd)
    random.shuffle(shuffled_tables)
    
    random.seed(sd)
    choice = [random.choice([True, False]) for _ in range(len(shuffled_text))]
    
    shuffled_report = []
    remaining_indices = list(range(len(shuffled_tables)))

    for i in range(len(shuffled_text)):
        # report should begin with text not table
        if i==0:
            shuffled_report.append(shuffled_text[i])
            continue
        if remaining_indices and choice.pop():
            index = remaining_indices.pop(0)
            shuffled_report.append(shuffled_tables[index])
        shuffled_report.append(shuffled_text[i])
    
    if remaining_indices:
        shuffled_report.extend(shuffled_tables[remaining_indices[0]:])

    return '\n'.join(shuffled_report), '\n'.join(shuffled_text), shuffled_tables


def main():

    args = parse_args()
    print(args)
    api = args.api
    model = args.model
    cache_dir = "../results/LLM_cache"
    lm_api = LanguageModelAPI(api, model, cache_dir)

    input_dir = "/data/SEC-10k/EXTRACTED_FILINGS_ITEMS_10K_WORDS"
    input_notable_dir = "/data/SEC-10k/EXTRACTED_FILINGS_ITEMS_NOTABLE/"
    output_base = args.output_base
    output_dir = f"{output_base}/{api}/{model}"
    os.makedirs(output_dir, exist_ok=True)

    file_paths = os.listdir(input_dir)
    np.random.seed(42)
    file_paths = np.random.choice(file_paths, 1000, replace=False)

    trunc_cnt = 0
    for file in tqdm.tqdm(file_paths, total=len(file_paths)):
        print(file)
        file_report = os.path.join(input_dir, file)
        file_notable = os.path.join(input_notable_dir, file)
        trunc_cnt += summarize_10k_varlen(args, lm_api, file_report, file_notable, output_dir)

    print('Summary finished. Total truncated report:', trunc_cnt)


if __name__ == "__main__":
    main()