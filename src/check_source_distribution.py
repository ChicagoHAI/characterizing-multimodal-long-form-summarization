import os
import numpy as np
import pandas as pd
import json
import re
import tqdm
import argparse
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, util
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from check_number_hallucination import NUMERIC_PATTERNS


def parse_args():

    parser = argparse.ArgumentParser(description='args for generating summary')
    parser.add_argument('--api', type=str, default='claude', 
                        choices=['claude', 'openai', 'cohere'], help='api type')
    parser.add_argument('--model', type=str, default='claude-2.1', 
                        choices=['gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'command', 'claude-2.0', 'claude-2.1'], help='model type')
    parser.add_argument('--shuffled', action='store_true', 
                        help='shuffle the report')
    parser.add_argument('--prompt', type=int, default=0,
                        choices=[0, 1], help='prompt version')
    parser.add_argument('--combined_num', type=int, default=1,
                        help='number of combined report sentences to be matched')
    parser.add_argument('--reshuffle_test', action='store_true',
                        help='get the source distribution of shuffled summary according to original report')
    parser.add_argument('--save_path', type=str, default='../results/summary_check', help='save path of analysis results')
    parser.add_argument('--root_path', type=str, default='../dataset/SEC-10k/LLM_SUMMERIZATION_ITEM7', help='root path of report and summary')
    args = parser.parse_args()

    args.root_path = f'../dataset/SEC-10k/LLM_SUMMERIZATION_ITEM7{"_SHUFFLED" if args.shuffled else ""}{"_P" if args.prompt!=0 else ""}{args.prompt if args.prompt!=0 else ""}/{args.api}/{args.model}'
    args.save_path = f'../results/summary_check/{args.api}/{args.model}/MiddleLoss/{"shuffled" if args.shuffled else "origin"}'

    return args


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


def greedy_match(s_sen, report_txt):

    stop_words = set(stopwords.words('english'))
    S = []
    F = {}
    for w in word_tokenize(s_sen):
        if w not in string.punctuation and w not in stop_words:
            if bool(re.search(r'\d', w)) and bool(re.search(r'[a-zA-Z]', w)) and len(re.findall(NUMERIC_PATTERNS, w))!=0:
                print(w)
                w1 = re.findall(NUMERIC_PATTERNS, w)[0]
                w2 = w.replace(w1, '')
                S.append(w1)
                S.append(w2)
            else:
                S.append(w)

    for r_sen in report_txt:
        i = 0
        while i < len(S):
            R = [w for w in word_tokenize(r_sen) if w not in string.punctuation and w not in stop_words]
            j = 0
            f_r = [] # longest matched token sequences of a specific report sentence
            
            while j < len(R):
                if S[i] == R[j]:
                    i_prime, j_prime = i, j
                    while i_prime < len(S) and j_prime < len(R) and S[i_prime] == R[j_prime]:
                        i_prime += 1
                        j_prime += 1
                    
                    if len(f_r) < i_prime-i:
                        f_r = S[i:i_prime]
                    
                    j = j_prime
                else:
                    j += 1
            
            if len(f_r) != 0:
                if r_sen in F:
                    F[r_sen].append(f_r)
                else:
                    F[r_sen] = [f_r]

            i = i + max(len(f_r), 1)

    return F


# Greedy match tokens & sentBERT for same token matched ratio to select top1
def find_similar_pairs(args, file, model):

    if os.path.exists(f'{args.save_path}/greedy_quotes/{file}'):
        return

    res_path0 = f'{args.save_path}/greedy_tokens_quotes'    # greedy matched tokens and score
    res_path1 = f'{args.save_path}/greedy_quotes'           # only for greedy match scores
    os.makedirs(res_path0, exist_ok=True)
    os.makedirs(res_path1, exist_ok=True)

    # Load report, text, tables, summary
    report, summary, tables, text = get_report_summary(file, args.root_path)
    
    summary_sen = []
    for sen in sent_tokenize(summary):
        summary_sen += sen.split('\n')
    summary_sen = [s.lower() for s in summary_sen if any(c.isalpha() for c in s)]
    report_txt = []
    for sen in sent_tokenize(text):
        report_txt += sen.split('\n')
    report_txt = [s.lower() for s in report_txt if s != '']
    report_txt = list(OrderedDict.fromkeys(report_txt)) # remove the duplicate sentences
    stop_words = set(stopwords.words('english'))

    greedy_match_tokens = {}
    greedy_match_score = {}

    # Greedy match tokens from each summary sentence with report sentences
    for s_sen in summary_sen:
        F = greedy_match(s_sen, report_txt)
        S = [w for w in word_tokenize(s_sen) if w not in string.punctuation and w not in stop_words]
        # Sort for top matched report sentences
        s_token_len = len(S)
        r_sen_extractive = []
        for r, fs in F.items(): # str: [[], [], ...]
            r_token_len = 0
            for f in fs:
                ####################
                # Add bonus for the token sequence length
                r_token_len += (len(f) + 0.1*len(f)*len(f))
                ####################
            extractive = r_token_len / s_token_len
            sen_extractive = tuple([r, extractive])
            r_sen_extractive.append(sen_extractive)
        
        r_sen_extractive = sorted(r_sen_extractive, key=lambda x:x[1], reverse=True)
        F_top10 = {}
        F_top10_score = {}

        # Use sentBERT embedding to choose the top match with same greedy match ratio
        if len(r_sen_extractive) > 0:
            sum_embeddings = model.encode(s_sen)
            
            top1_extractive = r_sen_extractive[0][-1]
            top1_embeddings = model.encode(r_sen_extractive[0][0])
            top1_cos = util.cos_sim(sum_embeddings, top1_embeddings)

            true_top1_index = 0
            for index, r_sen in enumerate(r_sen_extractive[:10]):
                if r_sen[1] == top1_extractive:
                    embeddings = model.encode(r_sen[0])
                    cos = util.cos_sim(sum_embeddings, embeddings)
                    if cos > top1_cos:
                        true_top1_index = index
                else:
                    break
            r_sen_extractive[0], r_sen_extractive[true_top1_index] = r_sen_extractive[true_top1_index], r_sen_extractive[0]

        for r_sen, extractive in r_sen_extractive[:10]:
            F[r_sen].append(extractive)
            F_top10[r_sen] = F[r_sen]
            F_top10_score[r_sen] = extractive

        greedy_match_tokens[s_sen] = F_top10
        greedy_match_score[s_sen] = F_top10_score

    json.dump(greedy_match_tokens, open(f'{res_path0}/{file}', 'w'))
    json.dump(greedy_match_score, open(f'{res_path1}/{file}', 'w'))


def find_similar_pairs_multisen(args, file, model):

    if os.path.exists(f'{args.save_path}/multi{args.combined_num}/greedy_quotes/{file}'):
        return

    res_path0 = f'{args.save_path}/multi{args.combined_num}/greedy_tokens_quotes'
    res_path1 = f'{args.save_path}/multi{args.combined_num}/greedy_quotes'
    os.makedirs(res_path0, exist_ok=True)
    os.makedirs(res_path1, exist_ok=True)
    
    report, summary, tables, text = get_report_summary(file, args.root_path)
    report_txt = []
    for sen in sent_tokenize(text):
        report_txt += sen.split('\n')
    report_txt = [s.lower() for s in report_txt if s != '']
    report_txt = list(OrderedDict.fromkeys(report_txt))
    stop_words = set(stopwords.words('english'))

    if args.combined_num > 2:
        ref_extr_sen = json.load(open(f'{args.save_path}/multi{args.combined_num-1}/greedy_tokens_quotes/{file}'))
    else:
        ref_extr_sen = json.load(open(f'{args.save_path}/greedy_tokens_quotes/{file}'))
    
    greedy_match_tokens = {}
    greedy_match_score = {} 

    for s_sen, r_sen_tokens in ref_extr_sen.items():
        if len(r_sen_tokens) == 0:
            continue
        
        top1_r_sen = list(r_sen_tokens.items())[0][0]
        top1_sim_score = list(r_sen_tokens.items())[0][1][-1]
        top1_matched_tokens_frag = list(r_sen_tokens.items())[0][1][:-1] # list of already matched tokens
        top1_matched_tokens = sum(top1_matched_tokens_frag, [])

        # Ignore summary sentence already has similarity score > 0.8
        if top1_sim_score >= 0.8:
            continue

        # Remove all of the already matched tokens
        s_sen_v2 = s_sen
        for word in top1_matched_tokens:
            s_sen_v2 = s_sen_v2.replace(word, '')
        
        F = greedy_match(s_sen_v2, report_txt)
        S = [w for w in word_tokenize(s_sen) if w not in string.punctuation and w not in stop_words]
        
        # Sort for top matched report sentences
        s_token_len = len(S)
        r_sen_extractive = []
        for r, fs in F.items(): # str: [[], [], ...]
            # Matched token length of combined report sentences
            r_token_len = len(top1_matched_tokens)
            for f in fs:
                r_token_len += (len(f) + 0.1*len(f)*len(f)) # Add bonus for the token sequence length
            extractive = r_token_len / s_token_len
            sen_extractive = tuple([r, extractive])
            r_sen_extractive.append(sen_extractive)
        
        r_sen_extractive = sorted(r_sen_extractive, key=lambda x:x[1], reverse=True)
        F_top10 = {}
        F_top10_score = {}

        # Use sentBERT embedding to choose the top match with same greedy match ratio
        if len(r_sen_extractive) > 0:
            sum_embeddings = model.encode(s_sen)
            
            top1_extractive = r_sen_extractive[0][-1]
            top1_embeddings = model.encode(r_sen_extractive[0][0])
            top1_cos = util.cos_sim(sum_embeddings, top1_embeddings)

            true_top1_index = 0
            for index, r_sen in enumerate(r_sen_extractive[:10]):
                if r_sen[1] == top1_extractive:
                    embeddings = model.encode(r_sen[0])
                    cos = util.cos_sim(sum_embeddings, embeddings)
                    if cos > top1_cos:
                        true_top1_index = index
                else:
                    break
            r_sen_extractive[0], r_sen_extractive[true_top1_index] = r_sen_extractive[true_top1_index], r_sen_extractive[0]

        for r, extractive in r_sen_extractive[:10]:
            F[r].extend(top1_matched_tokens_frag)
            F[r].append(extractive)
            r_comb = top1_r_sen + '\n' + r
            F_top10[r_comb] = F[r]
            F_top10_score[r_comb] = extractive
        
        if len(r_sen_extractive) == 0:
            F_top10[top1_r_sen+'\n'] = list(r_sen_tokens.items())[0][1]
            F_top10_score[top1_r_sen+'\n'] = top1_sim_score

        greedy_match_tokens[s_sen] = F_top10
        greedy_match_score[s_sen] = F_top10_score
        
    json.dump(greedy_match_tokens, open(f'{res_path0}/{file}', 'w'))
    json.dump(greedy_match_score, open(f'{res_path1}/{file}', 'w'))


def get_quotes_pos(args, file, top):

    if args.combined_num != 1:
        root_path = f'{args.save_path}/multi{args.combined_num}/greedy_quotes'
    else:
        root_path = f'{args.save_path}/greedy_quotes'

    if args.shuffled and args.reshuffle_test:
        file_path = args.root_path.replace('_SHUFFLED', '')
        print('-'*20)
        print(file_path)
    else:
        file_path = args.root_path
    report, summary, tables, text = get_report_summary(file, file_path)
    report_txt = []
    for sen in sent_tokenize(text):
        report_txt += sen.split('\n')
    report_txt = [s.lower() for s in report_txt if s != '']
    pairs = json.load(open(f'{root_path}/{file}', 'r'))
    
    sim_score_cat = [0]*11
    categories = [i/10 for i in range(1,11)]

    quote_distribution = [[0 for _ in range(5)] for _ in range(4)]

    for summary_sen, report_dict in pairs.items():
        for report_sen, score in list(report_dict.items())[:top]:
            flag = False
            # Count vs. Similarity score
            for i, cat in enumerate(categories):
                if round(cat-0.1, 1) < score <= cat:
                    sim_score_cat[i] += 1
                    flag = True
                    break
            if not flag:
                sim_score_cat[10] += 1
            
            # Position distribution
            if score > 0.7:
                multi_report_sen = report_sen.split('\n')[:args.combined_num]
                for sen in multi_report_sen:
                    try:
                        index = report_txt.index(sen)
                    except:
                        print(report_sen)
                        print(multi_report_sen)
                        print('-'*20)
                        continue
                    pos = index/len(report_txt)*100
                    pos_index = int(pos/20)
                    # pos_cat_sim[pos_index] += 1
                    if score > 1:
                        for i in range(4):
                            quote_distribution[i][pos_index] += 1
                    elif score > 0.9:
                        for i in range(3):
                            quote_distribution[i][pos_index] += 1
                    elif score > 0.8:
                        for i in range(2):
                            quote_distribution[i][pos_index] += 1
                    elif score > 0.7:
                        quote_distribution[0][pos_index] += 1

    return sim_score_cat, quote_distribution


def quotes_statistics(args):

    if args.combined_num == 1:
        save_path = args.save_path
    else:
        save_path = f'{args.save_path}/multi{args.combined_num}'
    file_paths = os.listdir(args.root_path)

    sim_score_top1 = [0]*11     # top 1 possible quotes similarity score distribution
    pos_distribution = [[0 for _ in range(5)] for _ in range(4)]

    for file in tqdm.tqdm(file_paths):
        print(file)
        sim_score_cat, pos_cat= get_quotes_pos(args, file, top=1)
        sim_score_top1 = [x+y for x, y in zip(sim_score_top1, sim_score_cat)]
        pos_distribution = [[pos_distribution[i][j] + pos_cat[i][j] for j in range(len(pos_cat[0]))] for i in range(len(pos_cat))]
    
    if args.shuffled and args.reshuffle_test:
        save_path1 = f'{save_path}/quotes_distribution_reshuffle.csv'
        save_path2 = f'{save_path}/similarity_score_reshuffle.csv'
    else:
        save_path1 = f'{save_path}/quotes_distribution.csv'
        save_path2 = f'{save_path}/similarity_score.csv'

    pos_distribution = np.array(pos_distribution)
    df1 = pd.DataFrame(pos_distribution)
    df1.to_csv(save_path1)

    sim_score_top1 = np.array(sim_score_top1)
    df0 = pd.DataFrame(sim_score_top1)
    df0.to_csv(save_path2)


def main():

    model = SentenceTransformer('all-MiniLM-L6-v2')
    args = parse_args()
    print(args)

    file_paths = os.listdir(args.root_path)

    for file in tqdm.tqdm(file_paths):
        print(file)
        if args.combined_num == 1:
            find_similar_pairs(args, file, model)
        else:
            find_similar_pairs_multisen(args, file, model)

    quotes_statistics(args)


if __name__ == '__main__':
    main()