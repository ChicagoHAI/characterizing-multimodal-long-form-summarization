import sys
sys.path.append('../')

import os
import numpy as np
import json
import re
import tqdm
import argparse
import webbrowser as web

from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from collections import OrderedDict

from utils.lm_api import LanguageModelAPI

NUMERIC_PATTERNS = r'(?<!\d)(?<![a-zA-Z-])\d{1,3}(?![a-jln-zA-JLN-Z\d])(?:,\d{3})*(?:\.\d+)?'

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

    return args


def check_number_hallucination(file: str, root_path: str, file_path: str):

    if os.path.exists(f'{root_path}/{file}'):
        return
    
    os.makedirs(root_path, exist_ok=True)

    num_hallucination_dict = {'A':{}, 'B':{}, 'C':{}, 'D':{}}

    report, summary, tables, text = get_report_summary(file, file_path)
    try:
    # get numbers in the summary
        _, _, summary_nums_dict, _, _ = get_ABCD_numbers(report, summary, tables, text)
    except:
        print('No legal numbers.')
        json.dump(num_hallucination_dict, open(f'{root_path}/{file}', 'w'))
        return

    # get summary sentences list, report text sentences list and tables list seperately
    summary_sen = []
    for sen in sent_tokenize(summary):
        summary_sen += sen.split('\n')
    summary_sen = [s.lower() for s in summary_sen if any(c.isalpha() for c in s)]

    report_txt = []
    for sen in sent_tokenize(text):
        report_txt += sen.split('\n')
    report_txt = [s.lower() for s in report_txt if s != '']
    report_txt = list(OrderedDict.fromkeys(report_txt)) # remove the duplicate sentences (you dare to believe that there are two exactly the same sentences in a report)

    report_tbl = re.split(r'(Table\s+\d+:)', tables)

    for num_type, num_dict in num_hallucination_dict.items():
        data = list(summary_nums_dict[num_type])
        if len(data) == 0:
            continue
        
        print('-'*20)
        print(f'{num_type} numbers')
        print('-'*20)

        flag = [0] * len(data)

        # get summary sentence that the number belongs to
        for a in data:
            for s in summary_sen:
                nums = re.findall(NUMERIC_PATTERNS, s)
                if a in nums:
                    num_dict[a] = [s]
                    break

        # traverse the report sentences list and extract relevant sentences (and the sentences right before and after it)
        for i in range(len(report_txt)):
            if i==0:
                relevant_sen = report_txt[i] + report_txt[i+1] + report_txt[i+2]
            elif i==len(report_txt)-1:
                relevant_sen = report_txt[i-2] + report_txt[i-1] + report_txt[i]
            else:
                relevant_sen = report_txt[i-1] + report_txt[i] + report_txt[i+1]

            # extract all the numbers from the sentence
            num_o = list(set(re.findall(NUMERIC_PATTERNS, report_txt[i])))

            # translate number words to digital numbers            
            number_words = extract_number_words(report_txt[i])
            for num_word, n in number_words.items():
                for j in range(len(data)):
                    if data[j] == n:
                        flag[j] = 1
                        # num_dict[data[j]].append(relevant_sen.replace(num_word, n)) # transfer the original number to certain format
                        num_dict[data[j]].append(relevant_sen)

            if len(num_o) == 0: # sentence contains no number
                continue

            # get different versions of the same number
            for n in num_o:
                n_formats = get_num_formats(n)
                for j in range(len(data)):
                    if data[j] in n_formats:
                        data_formatted = change_num_format(n_formats[0], data[j])
                        flag[j] = 1
                        # num_dict[data[j]].append(relevant_sen.replace(n, data_formatted))
                        num_dict[data[j]].append(relevant_sen)

        # traverse the report tables list and extract relevant tables (the whole tables)
        for i in range(2, len(report_tbl), 2):
            tbl_num_dict = {}
            num_o = re.findall(NUMERIC_PATTERNS, report_tbl[i])
            if len(num_o) == 0: # table contains no number
                continue

            for n in num_o:
                tbl_num_dict[n] = get_num_formats(n)
            for j in range(len(data)):
                j_tbl_num = [a for a, a_formats in tbl_num_dict.items() if data[j] in a_formats]
                if len(j_tbl_num) != 0:
                    flag[j] = 1
                    tbl = report_tbl[i]
                    for a in j_tbl_num:
                        data_formatted = change_num_format(a, data[j])
                        # tbl = tbl.replace(a, data_formatted)
                        tbl = tbl
                    num_dict[data[j]].append(report_tbl[i-1]+tbl) # Table index and table content
        
        # number that is not found in the text or tables
        for i in range(len(flag)):
            if flag[i] == 0:
                print(data[i], 'do not exist in the report')
                num_dict[data[i]].append('HALLUCINATION!')

        # # use GPT-4 to check hallucination
        # for num, sentences in num_dict.items():
        #     chat_output = chat_number_hallucination(num, sentences)
        #     print(chat_output)
        #     num_dict[num].append(chat_output)
        #     res = re.findall(r'[\[\<](.*?)[\]\>]', chat_output)[0] # a list of 'NO' or single 'YES'
        #     if res == 'NO':
        #         print(num, ':', sentences[0])
        #         print(chat_output)

        # # use Claude-3 to check hallucination
        # for num, sentences in num_dict.items():
        #     chat_output = chat_number_hallucination_claude(num, sentences, report)
        #     print(chat_output)
        #     num_dict[num].append(chat_output)
        
    # save num_hallucination_dict
    json.dump(num_hallucination_dict, open(f'{root_path}/{file}', 'w'))


def hallu_annotation(root_folder: str, report_path: str):
    
    file_path = f'{root_folder}/ABCD_nums'
    file_list = os.listdir(file_path)

    res_path = f'{root_folder}/ABCD_nums_label'
    os.makedirs(res_path, exist_ok=True)

    for file in file_list:
        if os.path.exists(f'{res_path}/{file}'):
            print(f'{file} existed, skipping ...')
            continue
        print('*'*20)
        print(file)
        print('*'*20)

        # open report web link
        report_dict = json.load(open(f'{report_path}/{file}', 'r'))
        report_link = report_dict['htm_filing_link']
        web.open(report_link)

        # print whole summary
        print(report_dict['summary'])

        num_report_dict = json.load(open(f'{file_path}/{file}', 'r'))
        num_labeled_dict = {'A':{}, 'B':{}, 'C':{}, 'D':{}}

        for num_type, num_dict in num_report_dict.items():
            # num: [summary_sentence, quote_0, quote_1, ..., quote_n]
            print('*'*5, num_type, 'numbers', '*'*5)
            for key, value in num_dict.items():
                print(key, ':', value[0])
                print('-'*10)

                for index, sen in enumerate(value[1:]):
                    print(index+1, repr(sen))
                    print()

                source = 0
                halu = 0
                other_halu = ''
                quote=''

                q = int(input_loop('exact source index (1,2,3...; 0: not find): '))
                print(q, end='\n\n')
                while True:
                    try:
                        if q!=0:
                            quote = value[q]
                        else:
                            quote = input('possible quote is: ')
                        break
                    except:
                        q = int(input_loop('exact source index (1,2,3...; 0: not find): '))
                        print(q, end='\n\n')


                source = int(input_loop('num source (1: text, 2: table, 3: text or table, 0: not find): '))
                print(source, end='\n\n')

                halu = int(input_loop('possible reason (1: fabricated number, 2: wrong operation, 3: rounding error, 4: time ambiguity, 5: correct operation, 6: context mismatch, 99: other types): '))
                print(halu, end='\n\n')

                if halu != 0:
                    other_halu = input('other info: ')
                    print(other_halu, end='\n\n')
                
                num_labeled_dict[num_type][key] = [value[0], quote, source, halu, other_halu]

        json.dump(num_labeled_dict, open(f'{res_path}/{file}', 'w'))


def input_loop(prompt: str):

    while True:
        a = input(prompt)
        if a.isdigit():
            value = int(a)
            break
        else:
            print("Reenter please: ", end='')

    return value


def ABCD_hallucination_statistics(root_folder):

    ABCD_type_dict = {1:'A', 2:'B', 3:'C', 0:'D'}
    halu_type_dict = {1: 'fabricated number', 2: 'wrong operation', 3: 'rounding error', 4: 'time ambiguity', 5: 'correct operation', 99: 'other types', 0: 'no hallucination'}
    
    ABCD_cnt = {}
    halu_cnt = {}
    
    for file in os.listdir(root_folder):
        ABCD_labels_dict = json.load(open(f'{root_folder}/{file}'))
        ABCD_nums = dict()
        for type, num_dict in ABCD_labels_dict.items():
            ABCD_nums.update(num_dict)
        
        for num, labels in ABCD_nums.items():
            ABCD_cnt[ABCD_type_dict[labels[2]]] = ABCD_cnt.get(ABCD_type_dict[labels[2]], 0) + 1 # ABCD types
            halu_cnt[halu_type_dict[labels[3]]] = halu_cnt.get(halu_type_dict[labels[3]], 0) + 1

    print(ABCD_cnt)
    print(halu_cnt)


number_words_dict = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"
}
# Function to extract number words from a sentence
def extract_number_words(sentence):

    words = sentence.lower().split()  # Convert to lowercase and split into words
    number_words = {word:number_words_dict[word] for word in words if word in list(number_words_dict.keys())}
    return number_words


def get_num_formats(data='23,255,300'):
    
    data_list = [data]
    
    if data.find(',') != -1:    # type 1: number with ','
        type = 1
    elif data.find('.') != -1:  # type 0: number only with '.'
        type = 0
    else:
        data_list.append(data + ',000')
        return data_list

    data_o = data
    if type == 1:
        data = data.replace('.', ',')       # 23,255.300 -> 23,255,300
        # 498,775 -> 0.5
        if data.find(',') == 3:
            data_list.append(str(round(float('0.'+data.replace(',', '')), 1)))
        data = data.replace(',', '.', 1)    # '23.255,300'
        data = data.replace(',', '')        # '23.255300'
    
    data = float(data)
    for i in range(1, 4):
        data_list.append(str(round(data, i)))

    data_list.append(str(round(data)))
    data_list.append(str(round(data)) + ',000')

    return data_list


def change_num_format(a:str, b:str):
    
    if a.find('.') != -1:
        a = a[:a.find('.')]
    a_numeric = ''.join(filter(str.isdigit, a))
    b_numeric = ''.join(filter(str.isdigit, b))
    b_numeric = b_numeric[1:] if b_numeric[0]=='0' else b_numeric
    len_a = len(a_numeric)
    len_b = len(b_numeric)
    if len_b < len_a:
        b_numeric = b_numeric + '0'*(len_a - len_b)
        b_formatted = '{:,}'.format(int(b_numeric))
        return b_formatted
    else:
        return b


def check_number_hallucination_gpt4(num: str, sentences: list):

    # use gpt-4 to check whether it could generate the number from the quotes
    api = 'openai'
    model = 'gpt-4'
    cache_dir = '../results/LLM_cache/'
    lm_api = LanguageModelAPI(api, model, cache_dir)    
    
    
    PROMPT1 = '''There are some quotes from a report between the <quote></quote> tags and a sentence between the <sentence></sentence> tags.

You should check whether you could find a certain numerical data between the <number></number> tags in the sentence from the quotes. Ignore the unit and rounding differences.\n\n'''
    PROMPT2 = 'Please begin your answer with \'[YES]\' if the data between the <number></number> tags in the sentence could be matched in the quotes or \'[NO]\' if you cannot. Then give me reasons for your judgment.'
    
    summary_pattern = r'(?<!\d{4})(?<![\.\w-])\d{1,3}(?![a-jln-zA-JLN-Z\d+])(?:,\d{3})*(?:\.\d+)?'
    sum_sentence = sentences[0]
    for m in re.finditer(summary_pattern, sum_sentence):
        if sum_sentence[m.start():m.end()] == num:
            sum_sentence = sum_sentence[:m.start()] + '<number>' + num + '</number>' + sum_sentence[m.end():]
    
    chat_input = PROMPT1 + 'Here are the quotes:\n\n<quotes>\n' + '\n'.join(sentences[1:]) + '\n</quotes>\n\nHere is the sentence:\n\n<sentence>\n' + sum_sentence + '\n</sentence>\n\n' + PROMPT2
       
    try:
        chat_output = lm_api.chat(chat_input)
    except:
        # If quotes are too long for the GPT-4, iterate all the quotes
        quotes_cnt = len(sentences[1:])
        num_lists = min(2, quotes_cnt) if api=='openai' else quotes_cnt
        quotes = sentences[1:]

        while num_lists <= quotes_cnt:
            step = quotes_cnt // num_lists
            quotes_list = [quotes[i:i+step] for i in range(0, len(quotes), step)]
            num_lists *= 2
            chat_output, chat_flag, except_quotes = chat_with_quotes(lm_api, quotes_list, sum_sentence)
            if chat_flag: # 1: yes, -1: no
                break
            else:
                quotes = []
                for i in except_quotes:
                    quotes += quotes_list[i]

    return chat_output

# Check numeric hallucination with report input to Claude and quotes generated by Claude
def check_number_hallucination_claude(file: str, root_path: str, file_path: str):

    os.makedirs(root_path, exist_ok=True)
    num_hallucination_dict = {'A':{}, 'B':{}, 'C':{}, 'D':{}}
    report, summary, tables, text = get_report_summary(file, file_path)

    # get numbers in the summary
    try:
        _, _, summary_nums_dict, _, _ = get_ABCD_numbers(report, summary, tables, text)
    except:
        print('No legal numbers.')
        json.dump(num_hallucination_dict, open(f'{root_path}/{file}', 'w'))
        return

    # get summary sentences
    summary_sen = []
    for sen in sent_tokenize(summary):
        summary_sen += sen.split('\n')
    summary_sen = [s for s in summary_sen if any(c.isalpha() for c in s)]

    for num_type, num_dict in num_hallucination_dict.items():
        data = list(summary_nums_dict[num_type])
        if len(data) == 0:
            continue
        
        print('-'*20)
        print(f'{num_type} numbers')
        print('-'*20)

        # get summary sentence that the number belongs to
        for a in data:
            for s in summary_sen:
                nums = re.findall(NUMERIC_PATTERNS, s)
                if a in nums:
                    num_dict[a] = [s]
                    break
        
        for num, sentences in num_dict.items():
            chat_output = chat_number_hallucination_claude(num, sentences[0], report)
            print(chat_output)
            num_dict[num].append(chat_output)

        json.dump(num_hallucination_dict, open(f'{root_path}/{file}', 'w'))


def chat_with_quotes(lm_api: LanguageModelAPI, quotes: list, sum_sentence: str):

    chat_flag = 0
    chat_output = []
    except_quotes = []

    PROMPT1 = '''There are some quotes from a report between the <quote></quote> tags and a sentence between the <sentence></sentence> tags.

You should check whether you could find a certain numerical data between the <number></number> tags in the sentence from the quotes. Ignore the unit and rounding differences.\n\n'''
    PROMPT2 = 'Please begin your answer with \'[YES]\' if the data between the <number></number> tags in the sentence could be matched in the quotes or \'[NO]\' if you cannot. Then give me reasons for your judgment.'
    # PROMPT2 = 'Please begin your answer with \'[YES]\' if the data between the <number></number> tags in the sentence could be matched in the quotes in the same context or \'[NO]\' if you cannot. Then give me reasons for your judgment.'

    for i in range(len(quotes)):
        chat_input = PROMPT1 + 'Here are the quotes:\n\n<quotes>\n' + '\n'.join(quotes[i]) + '\n</quotes>\n\nHere is the sentence:\n\n<sentence>\n' + sum_sentence + '\n</sentence>\n\n' + PROMPT2
        try:
            out = lm_api.chat(chat_input)
        except:
            except_quotes.append(i)
            continue
        chat_output.append(out)

        res = re.findall(r'\[(.*?)\]', chat_output[-1])[0]
        if res == 'YES':
            chat_flag = 1
            chat_output = chat_output[-1]
            break
    
    # yes
    if chat_flag == 1:
        return chat_output, 1, []
    # no or exception
    # return too long quotes that cause exception
    elif len(except_quotes) != 0:
        return '-'.join(chat_output), 0, except_quotes
    else:
        return '-'.join(chat_output), -1, []


def chat_number_hallucination_claude(num: str, sentences: str, report: str):

    api = 'claude'
    model = 'claude-3-haiku-20240307'
    cache_dir = '../results/LLM_cache/'
    lm_api = LanguageModelAPI(api, model, cache_dir)

    s_sen = sentences[0]
    # patterns = r'(?<!\d)(?<![a-zA-Z-])\d{1,3}(?![a-jln-zA-JLN-Z\d])(?:,\d{3})*(?:\.\d+)?' 
    for m in re.finditer(NUMERIC_PATTERNS, s_sen):
        if s_sen[m.start():m.end()] == num:
            s_sen = s_sen[:m.start()] + '<number>' + num + '</number>' + s_sen[m.end():]

            
    SYSTEM1 = 'You are an expert research assistant. Here is a document you will answer questions about: \n<doc>\n' # +report
    SYSTEM2 = '\n</doc>\n\nFirst, find the quotes from the document that are most relevant to the number between <number></number> tags in the given sentence, and then print them in numbered order. Quotes should be relatively short.  \n  \nIf there are no relevant quotes, write "No relevant quotes" instead.  \n  \nThen, starting with "Categorization:", judge the type of numeric hallucination.\n\nThere are five types of hallucination:\n' # + HALLUCINATION
    SYSTEM3 = '\n\nSimply give the name of the hallucination.\n\nIf there is no hallucination, say so.\n\nThen, explain the reasons for your judgement, starting with "Reasons:". Make references to quotes relevant to your reasons solely by adding their bracketed numbers at the end of relevant sentences.\n\nThus, the format of your overall response should look like what\'s shown between the <exmaple></example> tags. Make sure to follow the formatting and spacing exactly.  \n<example>  \nQuotes:  \n[1] Company X reported revenue of $12 million in 2021.  \n[2] Almost 90% of revene came from widget sales, with gadget sales making up the remaining 10%.  \n  \nCategorization: Terms Error. \n \nReasons: \nCompany X\'s revenue is $12 million. [1] Revenue is not net income.\n</example>'

    # SYSTEM1_2 = 'You are an expert research assistant. Here is a list of quotes you will refer to: \n<quotes>\n'
    # SYSTEM2_2 = '\n</quotes>\n\nYou will be given a sentence with a specific number between <number></number> tags. Starting with "Categorization:", judge the type of numeric hallucination of this specific number.\n\nThere are five types of hallucination:\n'
    # SYSTEM_3_2 = '\n\nSimply give the name of the hallucination.\n\nIf there is no hallucination, say so.\n\nThen, explain the reasons for your judgement, starting with "Reasons:". Make references to quotes relevant to your reasons solely by adding their bracketed numbers at the end of relevant sentences.\n\nThus, the format of your overall response should look like what\'s shown between the <exmaple></example> tags. Make sure to follow the formatting and spacing exactly.  \n<example>  \nCategorization: Terms Error. \n \nReasons: \nCompany X\'s revenue is $12 million. [1] Revenue is not net income.\n</example>'

    HALLUCINATION = '''Fabricated Number: Inaccurately stated compared with the value of the same entity in the report.
Context Mismatch: Discrepancy in terminology or language used between a report and its summary for the same number.
Rounding Error: Discrepancy arising from the rounding off of numeric values.
Arithmetic Error: Numbers generated from operations while the result of the operation is wrong.'''

    
    quotes = ''
    for i, s in enumerate(sentences[1:]):
        quotes += ('[' + str(i+1) + '] ' + s + '\n')

    system_prompt = SYSTEM1 + report + SYSTEM2 + HALLUCINATION + SYSTEM3 # mini
    # system_prompt = SYSTEM1_2 + quotes + SYSTEM2_2 + HALLUCINATION + SYSTEM_3_2
    user_prompt = s_sen

    print('-'*20)
    print(num, ':', s_sen)
    print('-'*20)

    chat_output = lm_api.chat(user_prompt, system_prompt)

    return chat_output
    

def get_report_summary(file: str, root_folder: str):
    
    file_path1 = f'{root_folder}/{file}'
    summary_combined = json.load(open(file_path1, 'r'))

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

    # Get report length and summary length, reset sum_length to multiples of 5
    summary_len = len(summary.split(' '))
    summary_len = ((summary_len-1) // 5) * 5 # summary length to 150, 155 ,160 ...

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

    print(repr(summary_v2))
    
    print('summary_nums:', summary_nums)
    for type, nums in report_nums_dict.items():
        if type == 'D':
            summary_nums_dict['D'] = summary_nums - summary_nums_dict['A'] - summary_nums_dict['B'] - summary_nums_dict['C']
        else:
            summary_nums_dict[type] = summary_nums.intersection(nums)
        print(type, ':', summary_nums_dict[type])
    
    return summary_len, report_nums_dict, summary_nums_dict, report_nums_cnt, summary_nums_cnt


def main():

    args = parse_args()
    print(args)
    file_path = f'{args.root_path}{"_SHUFFLED" if args.shuffled else ""}{"_P" if args.prompt!=0 else ""}{args.prompt if args.prompt!=0 else ""}/{args.api}/{args.model}'
    save_path = f'{args.save_path}/{args.api}/{args.model}/NumberHallucination/{"shuffled" if args.shuffled else "origin"}' + (f'/P{args.prompt}' if args.prompt!=1 and args.prompt!=0 else '')
    
    file_list = os.listdir(file_path)
    np.random.seed(42)
    file_list = np.random.choice(file_list, 100, replace=False)

    for file in tqdm.tqdm(file_list):
        print(file)
        check_number_hallucination(file, f'{save_path}/ABCD_nums', file_path)
    
    hallu_annotation(save_path, file_path)

    ABCD_hallucination_statistics(save_path)


if __name__ == '__main__':
    main()