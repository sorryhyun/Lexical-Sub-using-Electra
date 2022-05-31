def comp(list1, list2):
    res = []
    for word1 in list1:
        for word2 in list2:
            if word1[0] == word2[0]:
                res.append([word1[0], word1[1]])
    return res

def check(list1, list2):
    for word1 in list1:
        if word1 in list2:
            return 1
    return 0

def avgF(list1):
    temp = 0
    for word in list1:
        temp += word[1]
    res = temp / len(list1)
    return res

class dic_sorter:
    def __init__(self, max=10):
        self.dict = {}

    def put_words(self, wordlist):
        for word in wordlist:
            if word not in self.dict:
                self.dict[word] = 1
            else:
                self.dict[word] = self.dict[word] + 1

    def eraseone(self):
        resdict = self.dict.copy()
        for word in self.dict.keys():
            if resdict[word] == 1:
                del resdict[word]
        self.dict = resdict

    def output(self, max=10):
        res = [x[0] for x in sorted(self.dict.items(), key=lambda x: x[1], reverse=True)]
        return res[:max]

class dict_loader:
    def __init__(self, dict_path, bert_or_w2v='bert', bert_tokenizer_path='', w2v_model_path = ''):
        self.bert_or_w2v = bert_or_w2v
        from transformers import ElectraTokenizer
        import gensim
        if bert_tokenizer_path != '':
            self.tokenizer = ElectraTokenizer.from_pretrained(bert_tokenizer_path)
        if w2v_model_path != '':
            self.w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
        self.dict_path = dict_path

    def checkvocab(self, word):
        if self.bert_or_w2v == 'w2v':
            if word in self.w2v_model.wv.index2word :
                return 1
            else: return 0

        else:
            try:
                self.tokenizer.vocab[word]
            except KeyError:
                return 0
            return 1

    def preprocess_output_dict(self):
        import json
        with open(self.dict_path, 'r') as f:
            tempvocab = json.load(f)
        vocab = tempvocab.copy()

        # 훈련된 모델의 vocab에 없는 단어는 정답 동의어 리스트에서 제거.
        # 일정 freq 미만일 경우 제거하는 알고리즘은... 나중에 알아볼게여
        for i in tempvocab.keys():
            templist = tempvocab[i].copy()
            reslist = templist.copy()

            # 한글자짜리 키는 제거
            if len(i) < 2:
                del vocab[i]
                continue

            # key에 대한 value의 리스트단어들이...
            for word in templist:
                # w2v모델이 학습하지 못한 단어라면 리스트에서 제거
                if self.checkvocab(word) == 0:
                    reslist.remove(word)
                    continue

                # 한글자일때도 제거
                if len(word) < 2:
                    reslist.remove(word)

                # key와 동일단어일 때도 제거
                if word == i:
                    reslist.remove(word)

            # 만약 value가 비어버렸다면 key 자체를 제거
            # 사전이 관측 못한 키도 제거
            if len(reslist) == 0:
                del vocab[i]
            else:
                # 아니면 중복제거 후 업데이트
                reslist = list(set(reslist))
                vocab[i] = reslist

        return vocab
