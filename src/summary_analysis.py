import os
import numpy as np
import pandas as pd
import json
import re
import tqdm
import argparse
from transformers import PreTrainedTokenizerFast
import tiktoken

from check_number_hallucination import NUMERIC_PATTERNS

encoding_openai = tiktoken.get_encoding('cl100k_base')
encoding_claude = PreTrainedTokenizerFast(tokenizer_file='../dataset/claude-v1-tokenization.json')
encoding = {
    'openai': encoding_openai,
    'claude': encoding_claude
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
    parser.add_argument('--save_path', type=str, default='../results/summary_check', help='save path of analysis results')
    parser.add_argument('--root_path', type=str, default='../dataset/SEC-10k/LLM_SUMMERIZATION_ITEM7', help='root path of report and summary')
    args = parser.parse_args()

    args.root_path = f'../dataset/SEC-10k/LLM_SUMMERIZATION_ITEM7{"_SHUFFLED" if args.shuffled else ""}{"_P" if args.prompt!=0 else ""}{args.prompt if args.prompt!=0 else ""}'

    return args


def check_number_statistic(root_folder: str, save_path: str):

    r_nums_cnt = []
    s_nums_cnt = []
    num_cnt_sts = []
    r_cnt_ABCD_dict = {'A':[], 'B':[], 'C':[], 'D':[0]}
    s_cnt_ABCD_dict = {'A':[], 'B':[], 'C':[], 'D':[]}
    zeronum_cnt = 0

    file_list = os.listdir(root_folder)

    for file in tqdm.tqdm(file_list):
        print(file)
        report, summary, tables, text = get_report_summary(file, root_folder)
        summary_len, report_nums_dict, summary_nums_dict, report_nums_cnt, summary_nums_cnt = get_ABCD_numbers(report, summary, tables, text)

        if summary_nums_cnt == 0:
            zeronum_cnt += 1

        # Get numbers count
        r_nums_cnt.append(report_nums_cnt)
        s_nums_cnt.append(summary_nums_cnt)

        # Get ABCD numbers count
        for type, nums in report_nums_dict.items():
            r_cnt_ABCD_dict[type].append(len(nums))
            s_cnt_ABCD_dict[type].append(len(summary_nums_dict[type]))
    
    print('-'*10, 'number of summary with zero numbers', '-'*10)
    print(zeronum_cnt)
    print('-'*20)

    # Get total average number count
    num_cnt_sts.append(['Total', 'A', 'B', 'C', 'D'])
    num_cnt_sts.append([np.mean(r_nums_cnt)])
    num_cnt_sts.append([np.mean(s_nums_cnt)])

    # Get ABCD average number count
    for type, nums in s_cnt_ABCD_dict.items():
        num_cnt_sts[1].append(np.mean(r_cnt_ABCD_dict[type]))
        num_cnt_sts[2].append(np.mean(nums))

    # Save average number count
    num_cnt_sts = np.array(num_cnt_sts)
    df = pd.DataFrame(num_cnt_sts)
    df.to_csv(f'{save_path}/number_statistics.csv')


def check_summary_length(root_folder, save_path):
    
    words_length = {'item_7':[], 'item_7_truncated':[], 'summary':[]}
    token_length = {'item_7':[], 'item_7_truncated':[], 'summary':[]}

    for file in tqdm.tqdm(os.listdir(root_folder)):
        file_path = f'{root_folder}/{file}'
        chunked_10k = json.load(open(file_path, 'r'))
        ori_report = chunked_10k['item_7']
        report = chunked_10k['item_7_truncated']
        summary = chunked_10k['summary']
        
        words_length['item_7'].append(len(ori_report.split(' ')))
        words_length['item_7_truncated'].append(len(report.split(' ')))
        words_length['summary'].append(len(summary.split(' ')))

        token_length['item_7'].append(len(encoding['claude'].encode(ori_report)))
        token_length['item_7_truncated'].append(len(encoding['claude'].encode(report)))
        token_length['summary'].append(len(encoding['claude'].encode(summary)))

    words_res = []
    words_res.append(['type', 'avg', 'std', 'min', 'max'])
    words_res.append(['item_7', np.mean(words_length['item_7']), np.std(words_length['item_7'], ddof=1), np.min(words_length['item_7']), np.max(words_length['item_7'])])
    words_res.append(['item_7_truncated', np.mean(words_length['item_7_truncated']), np.std(words_length['item_7_truncated'], ddof=1), np.min(words_length['item_7_truncated']), np.max(words_length['item_7_truncated'])])
    words_res.append(['summary', np.mean(words_length['summary']), np.std(words_length['summary'], ddof=1), np.min(words_length['summary']), np.max(words_length['summary'])])
    words_res = np.array(words_res)
    df = pd.DataFrame(words_res)
    df.to_csv(f'{save_path}/summary_length_words.csv')

    token_res = []
    token_res.append(['type', 'avg', 'std', 'min', 'max'])
    token_res.append(['item_7', np.mean(token_length['item_7']), np.std(token_length['item_7'], ddof=1), np.min(token_length['item_7']), np.max(token_length['item_7'])])
    token_res.append(['item_7_truncated', np.mean(token_length['item_7_truncated']), np.std(token_length['item_7_truncated'], ddof=1), np.min(token_length['item_7_truncated']), np.max(token_length['item_7_truncated'])])
    token_res.append(['summary', np.mean(token_length['summary']), np.std(token_length['summary'], ddof=1), np.min(token_length['summary']), np.max(token_length['summary'])])
    token_res = np.array(token_res)
    df = pd.DataFrame(token_res)
    df.to_csv(f'{save_path}/summary_length_token.csv')


def get_report_summary(file: str,
                       root_folder: str):
    
    file_path = f'{root_folder}/{file}'
    summary_combined = json.load(open(file_path, 'r'))

    # get report, summary, tables, text
    report = summary_combined['item_7_truncated']
    try:
        summary = summary_combined['summary']
    except:
        summary = summary_combined['10000']

    tables = summary_combined['item_7_tables']
    text = summary_combined['item_7_text']
    if len(tables)==0:
        tables = 'No tables.'

    return report, summary, tables, text


def get_ABCD_numbers(report: str, summary: str, tables: str, text: str):

    summary_len = len(summary.split(' '))

    report_nums_dict = {}
    summary_nums_dict = {}
    
    # exclude all dates number
    patterns_date = r'(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},'
    report_v2 = re.sub(patterns_date, '', report)
    text_v2 = re.sub(patterns_date, '', text)
    summary_v2 = re.sub(patterns_date, '', summary)
    # exclude table index
    patterns_idx = r'Table \d+:'
    report_v2 = re.sub(patterns_idx, '', report_v2)
    tables_v2 = re.sub(patterns_idx, '', tables)

    # numbers from the report
    report_nums_all = re.findall(NUMERIC_PATTERNS, report_v2)
    report_nums = set(report_nums_all)
    report_nums_cnt = len(report_nums)
    # numbers in table
    table_nums_all = re.findall(NUMERIC_PATTERNS, tables_v2)
    report_nums_dict['B'] = set(table_nums_all)
    # numbers in text
    text_nums_all = re.findall(NUMERIC_PATTERNS, text_v2)
    report_nums_dict['A'] = set(text_nums_all)
    
    # numbers in both table and text (C)
    report_nums_dict['C'] = report_nums_dict['B'].intersection(report_nums_dict['A'])
    # numbers only in table (B)
    report_nums_dict['B'] = report_nums_dict['B'] - report_nums_dict['C']
    # numbers only in text (A)
    report_nums_dict['A'] = report_nums_dict['A'] - report_nums_dict['C']
    # numbers not in the report (D)
    report_nums_dict['D'] = {}

    # numbers from the summary
    summary_nums_all = re.findall(NUMERIC_PATTERNS, repr(summary_v2))
    summary_nums = set(summary_nums_all)
    summary_nums_cnt = len(summary_nums)

    print('summary_nums:', summary_nums)
    for type, nums in report_nums_dict.items():
        if type == 'D':
            summary_nums_dict['D'] = summary_nums - summary_nums_dict['A'] - summary_nums_dict['B'] - summary_nums_dict['C']
        else:
            summary_nums_dict[type] = summary_nums.intersection(nums)
        print(type, ':', summary_nums_dict[type])
    
    return summary_len, report_nums_dict, summary_nums_dict, report_nums_cnt, summary_nums_cnt

# There are many corner-cases in the tables that have text data instead of numerical
def extract_true_tables(root_folder1='../dataset/SEC-10k/LLM_SUMMERIZATION_ITEM7/claude/claude-2.0',
                        root_folder2='/data/SEC-10k/EXTRACTED_FILINGS_ITEMS_NOTABLE'):
    
    for file in tqdm.tqdm(os.listdir(root_folder1)):
        print(file)
        res_dict = json.load(open(f'{root_folder1}/{file}', 'r'))
        ori_text = json.load(open(f'{root_folder2}/{file}', 'r'))['item_7']

        report = res_dict['item_7']
        # truncated report
        report = " ".join(report.split(" ")[:10000])
        # truncated text
        text = report

        # tables
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

        for i in range(len(tables)):
            print(repr(tables[i]))
            print('-'*10)
            print(repr(clean_tables[i]))
            print('='*20)

        true_tables = []
        for i in range(len(clean_tables)):
            # false table: replace the false table in the report and text to clean table without table tags
            if ori_text.find(clean_tables[i][6:-6]) != -1 or ori_text.find(clean_tables[i][-80:-6]) != -1:
                print('#'*10)
                print(tables[i])
                print('#'*10)
                report = report.replace(tables[i], clean_tables[i])
                text = text.replace(tables[i], clean_tables[i])
                continue
            # true tables
            else:
                if text.find(tables[i]) != -1:
                    true_tables.append(tables[i])
                    text = text.replace(tables[i], '')
                else:
                    print('$'*10)
                    print(repr(tables[i]))
                    print('$'*10)


        res_dict['item_7_tables'] = ''.join(true_tables)
        res_dict['item_7_text'] = text
        res_dict['item_7_truncated'] = report

        json.dump(res_dict, open(f'{root_folder1}/{file}', 'w'))


def main():
    
    args = parse_args()
    print(args)

    root_folder = f'{args.root_path}/{args.api}/{args.model}'
    res_path = f'{args.save_path}/{args.api}/{args.model}/NumberStatistics/' + ('shuffled' if args.shuffled else 'origin') + (f'/P{args.prompt}' if args.prompt!=1 and args.prompt!=0 else '')
    os.makedirs(res_path, exist_ok=True)

    check_summary_length(root_folder, res_path)
    check_number_statistic(root_folder, res_path)


if __name__ == "__main__":
    main()