import os, json

paragraph_list = []
sample_size = 20

def run_metric(res : []):
    Precision = [0 for _ in range(10)]
    Recall = [0 for _ in range(10)]

    n = 0
    for n, data in enumerate(res):

        if len(data['inference_list']) == 0: continue
        temp = [data['inference_list'][:i + 1] for i in range(10)]

        for i in range(10):
            for word in temp[i]:
                if word in data['answer_list']:
                    Precision[i] += 1 / len(temp[i])
                    Recall[i] += 1 / len(data['answer_list'])

    Precision = [round(x / n, 4) for x in Precision]
    Recall = [round(x / n, 4) for x in Recall]
    F1score = [round(2 * x * y / (x + y), 4) for x, y in zip(Precision, Recall)]
    return {'Precision' : Precision,
            'Recall':Recall,
            'F1score': F1score}

import re
def preprocess_sentence(sentence):
    res = re.sub('[「＜『《]', '(', sentence)
    res = re.sub('[＞」』》]', ')', res)
    res = re.sub('[│ㆍ]', ', ', res)
    res = re.sub('[^ \-<>/,.()!?%$\'\":;+a-zA-Zㄱ-ㅣ가-힣0-9]', '', res)
    res = re.sub('\\\\\'', '', res)
    res = res.lstrip()
    return res

def sample_paragraphs(target_word):
    import numpy as np
    index_list = np.random.choice(len(paragraph_list), len(paragraph_list), replace=False).tolist()

    target_paragraph_list = []
    for i in index_list:
        sentence = paragraph_list[i]
        if sentence.startswith(target_word) == True:
            preprocessed_sentence = preprocess_sentence(sentence)
            target_paragraph_list.append([preprocessed_sentence, 0])
        elif sentence.find(' ' + target_word) != -1:
            preprocessed_sentence = preprocess_sentence(sentence)
            word_index = preprocessed_sentence.find(' ' + target_word)
            target_paragraph_list.append([preprocessed_sentence, word_index])
        else : continue
        if len(target_paragraph_list) == sample_size : break

    return {'target_word': target_word, 'target_paragraph_list': target_paragraph_list}

class tester:
    def __init__(self, test_type, preprocessed_corpus_path, sample_sizeHP=20, worker=8):
        self.exp_result = {}
        self.test_type = test_type
        global paragraph_list
        with open(preprocessed_corpus_path, 'r') as f:
            paragraph_list = json.load(f)
        global sample_size
        sample_size = sample_sizeHP
        self.worker = worker

        import Model
        from utils import dict_loader
        if self.test_type == 'w2v':
            self.model = Model.word2vecmodel(w2v_model_path = './data/w2v.model')
            self.dict = dict_loader(dict_path='./data/4paper.json', bert_or_w2v='w2v',
                                    w2v_model_path = './data/vocab_check.model').preprocess_output_dict()
        else:
            self.model = Model.LS_models()
            self.dict = dict_loader(dict_path='./data/4paper.json', bert_or_w2v='bert',
                                    bert_tokenizer_path="monologg/koelectra-large-v3-generator").preprocess_output_dict()

    def test(self, rep=3):
        from utils import check
        import Model
        from tqdm import tqdm
        inferenced_list = []

        if self.test_type == 'w2v':
            self.model : Model.word2vecmodel

            # 동의어 사전의 모든 단어를 참조하므로 keys()
            for n, target_word in enumerate(tqdm(self.dict.keys())):

                # 정답단어 리스트 추출 코드
                answerlist = self.dict.get(target_word)

                # 모델의 추론 리스트 추출 코드.
                inferencelist = self.model.inference(target_word)

                # 정답리스트는 과연 몇등일까?
                inferenced_list.append({'inference_list':inferencelist,
                                        'answer_list':answerlist,
                                        'tp':check(answerlist, inferencelist)
                })
            res = run_metric(inferenced_list)
            with open('./' + self.test_type + '_result.json', 'w') as f:
                json.dump(res, f, indent=4)

        else:
            bert_res = {'Precision@1':0, 'F1score@5':0, 'F1score@10':0}
            for reped in range(rep):
                self.model: Model.LS_models
                import re
                print('sampling sentences from corpus...')
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                # 코퍼스에서 단어가 등장하는 n개의 문단 추출
                # 문장과 문장 내 단어의 인덱스를 검색
                import multiprocessing
                with multiprocessing.Pool(self.worker) as p:
                    sampled_data = list(tqdm(p.imap(sample_paragraphs, list(self.dict.keys())), total=len(self.dict.keys())))
                p.close()
                p.join()

                os.environ['TOKENIZERS_PARALLELISM'] = 'true'
                input_target = []
                for data in sampled_data:
                    data['answer_list'] = self.dict[data['target_word']]
                    input_target.append(data)

                # 모델에 입력
                print('operating Lexical substitution...')
                for data in tqdm(input_target):
                    if self.test_type == 'bert':
                        data['inference_list'], _ = \
                            self.model.inference_bert(data['target_paragraph_list'], data['target_word'])
                    elif self.test_type == 'bert-ls':
                        data['inference_list'], _ = \
                            self.model.inference_BERTLS(data['target_paragraph_list'], data['target_word'])
                    elif self.test_type == 'electra':
                        data['inference_list'], _ = \
                            self.model.inference_electra(data['target_paragraph_list'], data['target_word'])
                    elif self.test_type == 'electra-ls':
                        data['inference_list'], _ = \
                            self.model.inference_BERTLS_with_electra(data['target_paragraph_list'], data['target_word'])
                    inferenced_list.append(data)

                    # 추출된 결과물 (단어 리스트) 를 단어사전에 대해 검증
                for n in range(len(inferenced_list)):
                    inferenced_list[n]['tp'] = check(inferenced_list[n]['answer_list'],
                                                     inferenced_list[n]['inference_list'])
                res = run_metric(inferenced_list)
                bert_res['Precision@1'] += res['Precision'][0]/rep
                bert_res['F1score@5'] += res['F1score'][4]/rep
                bert_res['F1score@10'] += res['F1score'][9]/rep
                with open('./' + self.test_type + '_result_' + str(reped) + '.json', 'w') as f:
                    json.dump(res, f, indent=4)

            with open('./' + self.test_type + '_result_3average.json', 'w') as f:
                json.dump(bert_res, f, indent=4)

    def inference(self, target_word, preprocessed_corpus_path='', sample_size=20):
        import Model
        if self.test_type == 'w2v':
            self.model: Model.word2vecmodel
            return self.model.inference(target_word)

        else:
            self.model: Model.LS_models
            if not os.path.isfile(preprocessed_corpus_path):
                print("No preprocessed corpus")
                return
            with open(preprocessed_corpus_path, 'r') as f:
                paragraph_list = json.load(f)

            # 코퍼스에서 단어가 등장하는 n개의 문단 추출
            templist = []
            for paragraph in paragraph_list:
                if paragraph.startswith(target_word) == True or paragraph.find(' ' + target_word) != -1:
                    templist.append(paragraph)

            import random
            if sample_size > len(templist):
                sampled_paragraph_list = templist
            else:
                sampled_paragraph_list = random.sample(templist, sample_size)

            # 문장과 문장 내 단어의 인덱스를
            target_paragraph_list = []
            for sampled_paragraph in sampled_paragraph_list:
                if sampled_paragraph.startswith(target_word) == True:
                    word_index = 0
                else:
                    word_index = sampled_paragraph.find(' ' + target_word)
                target_paragraph_list.append([sampled_paragraph, word_index])

            # 모델에 입력 (100개의 문단 리스트, 타겟 단어)
            inferencelist = ''
            if self.test_type == 'bert':
                inferencelist, _ = self.model.inference_bert(target_paragraph_list, target_word)
            elif self.test_type == 'bert-ls':
                inferencelist, _ = self.model.inference_BERTLS(target_paragraph_list, target_word)
            elif self.test_type == 'electra':
                inferencelist, _ = self.model.inference_electra(target_paragraph_list, target_word)
            elif self.test_type == 'electra-ls':
                inferencelist, _ = self.model.inference_BERTLS_with_electra(target_paragraph_list, target_word)

            return inferencelist
