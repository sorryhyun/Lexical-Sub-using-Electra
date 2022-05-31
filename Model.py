import gensim
import re
from utils import comp, avgF, dic_sorter


class word2vecmodel:
    def __init__(self, w2v_model_path, list_num=10):
        self.w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
        self.list_num = list_num
        self.cache = {}

    def check_target_rank(self, input, answer):
        templist = self.w2v_model.wv.most_similar(input, topn=len(self.w2v_model.wv.index2word))
        for n, word in enumerate(templist):
            if word[0] == answer:
                n += 1
                return n
        return -1

    def inference(self, input, thres_tokens=1):
        result = self.w2v_model.wv.most_similar(input, topn=self.list_num*2)

        for word in result:
            temp = self.cache.get(word[0], False)
            if temp is False:
                templist = self.w2v_model.wv.most_similar(word[0], topn=self.list_num)
                self.cache[word[0]] = templist
            else :
                templist = temp

            temp = comp(templist, result)
            if len(temp) < thres_tokens:
                result.remove(word)

        res = []
        for word in result:
            if len(word[0]) > 1:
                res.append(word[0])

        return res[:10]

    def return_count(self, input):
        try:
            return self.w2v_model.wv.vocab[input].count
        except KeyError:
            return -1

    def simple_inference(self, input, topn=10):
        result = self.w2v_model.wv.most_similar(input, topn=(topn+10))

        # for wordpiece
        for word in result:
            if '##' in word[0] or len(word[0]) < 2:
                result.remove(word)
        result = [word[0] for word in result]

        return result[:10]


class Electra_model:
    def __init__(self, discriminator_path):
        from transformers import ElectraTokenizerFast, ElectraForPreTraining
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(discriminator_path)
        self.disc_model = ElectraForPreTraining.from_pretrained(discriminator_path).cuda()

    def check_by_electra(self, inputs, target_index):
        import torch
        inputs = inputs.type(torch.int64).cuda()
        output = self.disc_model(inputs)
        return output.logits[:, target_index].tolist()


class LS_models:
    def __init__(self, generator_path = "monologg/koelectra-large-v3-generator",
                 discriminator_path = "monologg/koelectra-large-v3-discriminator"):
        from transformers import ElectraTokenizerFast, ElectraForMaskedLM
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert_model = ElectraForMaskedLM.from_pretrained(generator_path).cuda()
        self.tokenizer = ElectraTokenizerFast.from_pretrained(generator_path)
        self.electra_model = Electra_model(discriminator_path)
        self.statemp = []

    # 토큰목록을 입력받아 토큰에 정제된 단어가 매핑된 딕셔너리를 출력한다.
    # ##이 붙어있거나 띄어쓰기로 분리된 토큰을 붙여준다
    # 또한 필요에 따라 target word와 동일한 단어목록을 제거한다.
    def token2word(self, tokens, target_word, erase_self=True):
        res = {}
        decoded_list = self.tokenizer.batch_decode(tokens)
        for decoded, token in zip(decoded_list, tokens):
            normalized_word = re.sub(' ', '', decoded)
            if erase_self == True:
                if normalized_word == target_word or len(normalized_word) < 2 or '#' in normalized_word:
                    continue
            else:
                if len(normalized_word) < 2 or '#' in normalized_word:
                    continue
            res[token] = normalized_word
        return res

    def check_res_with_electra(self, token_words, inputs, target_token_index):
        if len(token_words.keys()) == 0 : return []

        # 문장의 목표 단어를 대체어 후보로 대치하기 위해 대체어 후보 갯수만큼 텐서 생성
        import torch
        input_tokens = torch.stack([inputs for _ in range(len(token_words))], dim=0).cuda()

        # 대체어 후보로 대치
        for n, token in enumerate(list(token_words.keys())):
            input_tokens[n][target_token_index] = token

        electra_res = self.electra_model.check_by_electra(input_tokens, target_token_index)

        for val, key in zip(electra_res, list(token_words.keys())):
            if val > 1.0:
                del token_words[key]
        return token_words

    def inference_bert(self, sentence_list, word):
        from torch.nn import functional as F
        import torch
        self.statemp = []
        dict = dic_sorter()

        # 해당 데이터셋의 랜덤 문장 / 모든 문장 에 대해서 수행
        target_token_indexes = []
        sentences = [x[0] for x in sentence_list]
        tokenized_sentences = self.tokenizer([x[0] for x in sentence_list])
        target_word_indexes = [x[1] for x in sentence_list]

        # 문장의 토큰 길이가 bert 용량을 넘으면 continue
        input_indexes=[]
        for n, tokens in enumerate(tokenized_sentences.data['input_ids']):
            if len(tokens) <= 512:
                input_indexes.append(n)

        # 타겟 토큰의 인덱스를 찾는 작업
        input_sentences = []
        for i in input_indexes:
            # 타겟 토큰의 인덱스를 찾는 작업
            if target_word_indexes[i] != 0:
                sentence_till_target_word = sentences[i][:target_word_indexes[i]]
                temp = self.tokenizer(sentence_till_target_word)
                target_token_index = len(temp['input_ids']) - 1
            else:
                target_token_index = 1
            input_sentences.append(sentences[i])
            target_token_indexes.append(target_token_index)

        input_batch = self.tokenizer(input_sentences, return_tensors="pt", padding='max_length')
        with torch.no_grad():
            input_batch.to(device=self.device)
            token_logits = self.bert_model(**input_batch).logits
            softmax = F.softmax(token_logits, dim=-1)
        for i in range(len(softmax)):
            mask_token_logits = softmax[i, target_token_indexes[i], :]
            bert_outputs = torch.topk(mask_token_logits, 10, dim=0).indices.tolist()
            res = self.token2word(bert_outputs, target_word=word)
            if len(res) == 0: continue
            dict.put_words(list(res.values())[:4])

        # 출력 결과물들을 취합하는 과정. 빈도수별로 출력하여 상위 10개 컷
        return dict.output(), self.statemp


    def inference_electra(self, sentence_list, word):
        from torch.nn import functional as F
        import torch
        self.statemp = []
        dict = dic_sorter()

        # 해당 데이터셋의 랜덤 문장 / 모든 문장 에 대해서 수행
        target_token_indexes = []
        sentences = [x[0] for x in sentence_list]
        tokenized_sentences = self.tokenizer([x[0] for x in sentence_list])
        target_word_indexes = [x[1] for x in sentence_list]

        # 문장의 토큰 길이가 bert 용량을 넘으면 continue
        input_indexes=[]
        for n, tokens in enumerate(tokenized_sentences.data['input_ids']):
            if len(tokens) <= 512:
                input_indexes.append(n)

        # 타겟 토큰의 인덱스를 찾는 작업
        input_sentences = []
        for i in input_indexes:
            # 타겟 토큰의 인덱스를 찾는 작업
            if target_word_indexes[i] != -1:
                sentence_till_target_word = sentences[i][:target_word_indexes[i]]
                temp = self.tokenizer(sentence_till_target_word)
                target_token_index = len(temp['input_ids']) - 1
            else:
                target_token_index = 1
            input_sentences.append(sentences[i])
            target_token_indexes.append(target_token_index)
        input_batch = self.tokenizer(input_sentences, return_tensors="pt", padding='max_length')
        with torch.no_grad():
            input_batch.to(device=self.device)
            token_logits = self.bert_model(**input_batch).logits
            softmax = F.softmax(token_logits, dim=-1)

        for i in range(len(softmax)):
            mask_token_logits = softmax[i, target_token_indexes[i], :]
            bert_outputs = torch.topk(mask_token_logits, 10, dim=0).indices.tolist()
            res = self.token2word(bert_outputs, target_word=word)
            res = self.check_res_with_electra(res, input_batch.data['input_ids'][i], target_token_indexes[i])
            if len(res) == 0:continue
            dict.put_words(list(res.values())[:4])
        # 출력 결과물들을 취합하는 과정. 빈도수별로 출력하여 상위 10개 컷
        return dict.output(), self.statemp


    def dropout_token(self, embeddings, target_token_indexes):
        import torch.nn.functional as F
        res = embeddings.clone().detach()
        for n, target_token_index in enumerate(target_token_indexes):
            embeddings_with_dropout = F.dropout(res[n][target_token_index], p=0.3, training=True)
            res[n][target_token_index] = embeddings_with_dropout
        return res

    def cal_eq1(self, logit_dict, original_token_logits):
        import torch
        for token in logit_dict.keys():
            logit_dict[token] = torch.tensor(0.01 * (torch.log(logit_dict[token])
                                        - torch.log(1 - original_token_logits)).item())

        return logit_dict

    def cal_eq2(self, temp_dict, input, target_token_index):
        import torch
        import torch.nn.functional as F
        subbed = input.clone().detach()
        original_hid = self.bert_model.electra(input, output_hidden_states=True)['hidden_states']
        original_att = self.bert_model.electra(input, output_attentions=True)['attentions']
        original_hids = torch.cat(original_hid[-12:-1], dim=2)

        original_att = torch.cat(original_att, dim=1).mean(dim=1)
        original_att = original_att[:, :, target_token_index]

        for token in temp_dict.keys():
            subbed[0][target_token_index] = token
            subbed_hid = self.bert_model.electra(subbed, output_hidden_states=True)['hidden_states']
            subbed_hids = torch.cat(subbed_hid[-12:-1], dim=2)
            similarities = F.cosine_similarity(original_hids, subbed_hids, dim=2)
            temp_dict[token] = temp_dict[token] + (original_att * similarities).sum().item()

        return temp_dict

    def inference_BERTLS(self, sentence_list, word):
        from torch.nn import functional as F
        import torch
        self.statemp = []
        dict = dic_sorter()

        # 해당 데이터셋의 랜덤 문장 / 모든 문장 에 대해서 수행
        target_token_indexes = []
        sentences = [x[0] for x in sentence_list]
        tokenized_sentences = self.tokenizer([x[0] for x in sentence_list])
        target_word_indexes = [x[1] for x in sentence_list]

        # 문장의 토큰 길이가 bert 용량을 넘으면 continue
        input_indexes = []
        for n, tokens in enumerate(tokenized_sentences.data['input_ids']):
            if len(tokens) <= 512:
                input_indexes.append(n)

        # 타겟 토큰의 인덱스를 찾는 작업
        input_sentences = []
        for i in input_indexes:
            # 타겟 토큰의 인덱스를 찾는 작업
            if target_word_indexes[i] != -1:
                sentence_till_target_word = sentences[i][:target_word_indexes[i]]
                temp = self.tokenizer(sentence_till_target_word)
                target_token_index = len(temp['input_ids']) - 1
            else:
                target_token_index = 1
            input_sentences.append(sentences[i])
            target_token_indexes.append(target_token_index)

        input_batch = self.tokenizer(input_sentences, return_tensors="pt", padding='max_length')
        with torch.no_grad():
            input_batch.to(device=self.device)
            embeddings = self.bert_model.electra.embeddings.word_embeddings(input_batch.data['input_ids'])
            input_embed = self.dropout_token(embeddings, target_token_indexes)
            token_logits = self.bert_model(inputs_embeds=input_embed).logits
            softmax = F.softmax(token_logits, dim=-1)

            for i in range(len(softmax)):
                target_token_logits = softmax[i, target_token_indexes[i], :]
                bert_outputs = torch.topk(target_token_logits, 50, dim=0).indices.tolist()
                original_token_logit = target_token_logits[bert_outputs[0]]
                temp_dict={}
                for token in bert_outputs:
                    if token == input_batch.data['input_ids'][i][target_token_indexes[i]]: continue
                    temp_dict[token] = target_token_logits[token].clone().detach()

                temp_dict = self.cal_eq1(temp_dict, original_token_logit)
                temp_dict = self.cal_eq2(temp_dict, input_batch.data['input_ids'][i].unsqueeze(0), target_token_indexes[i])

                bertls_outputs = []
                for tups in sorted(temp_dict.items(), key=lambda x: x[1], reverse=True):
                    bertls_outputs.append(tups[0])

                res = self.token2word(bertls_outputs, target_word=word)
                if len(res) == 0: continue
                dict.put_words(list(res.values())[:4])

        # 출력 결과물들을 취합하는 과정. 빈도수별로 출력하여 상위 10개 컷
        return dict.output(), self.statemp


    def inference_BERTLS_with_electra(self, sentence_list, word):
        from torch.nn import functional as F
        import torch
        self.statemp = []
        dict = dic_sorter()

        # 해당 데이터셋의 랜덤 문장 / 모든 문장 에 대해서 수행
        target_token_indexes = []
        sentences = [x[0] for x in sentence_list]
        tokenized_sentences = self.tokenizer([x[0] for x in sentence_list])
        target_word_indexes = [x[1] for x in sentence_list]

        # 문장의 토큰 길이가 bert 용량을 넘으면 continue
        input_indexes = []
        for n, tokens in enumerate(tokenized_sentences.data['input_ids']):
            if len(tokens) <= 512:
                input_indexes.append(n)

        # 타겟 토큰의 인덱스를 찾는 작업
        input_sentences = []
        for i in input_indexes:
            # 타겟 토큰의 인덱스를 찾는 작업
            if target_word_indexes[i] != -1:
                sentence_till_target_word = sentences[i][:target_word_indexes[i]]
                temp = self.tokenizer(sentence_till_target_word)
                target_token_index = len(temp['input_ids']) - 1
            else:
                target_token_index = 1
            input_sentences.append(sentences[i])
            target_token_indexes.append(target_token_index)

        input_batch = self.tokenizer(input_sentences, return_tensors="pt", padding='max_length')
        with torch.no_grad():
            input_batch.to(device=self.device)
            embeddings = self.bert_model.electra.embeddings.word_embeddings(input_batch.data['input_ids'])
            input_embed = self.dropout_token(embeddings, target_token_indexes)
            token_logits = self.bert_model(inputs_embeds=input_embed).logits
            softmax = F.softmax(token_logits, dim=-1)

            for i in range(len(softmax)):
                target_token_logits = softmax[i, target_token_indexes[i], :]
                bert_outputs = torch.topk(target_token_logits, 50, dim=0).indices.tolist()
                original_token_logit = target_token_logits[bert_outputs[0]]
                temp_dict = {}
                for token in bert_outputs:
                    if token == input_batch.data['input_ids'][i][target_token_indexes[i]]: continue
                    temp_dict[token] = target_token_logits[token].clone().detach()

                temp_dict = self.cal_eq1(temp_dict, original_token_logit)
                temp_dict = self.cal_eq2(temp_dict, input_batch.data['input_ids'][i].unsqueeze(0), target_token_indexes[i])

                bertls_outputs = []
                for tups in sorted(temp_dict.items(), key=lambda x: x[1], reverse=True):
                    bertls_outputs.append(tups[0])

                res = self.token2word(bertls_outputs, target_word=word)
                res = self.check_res_with_electra(res, input_batch.data['input_ids'][i], target_token_indexes[i])
                if len(res) == 0: continue
                dict.put_words(list(res.values())[:4])

        # 출력 결과물들을 취합하는 과정. 빈도수별로 출력하여 상위 10개 컷
        return dict.output(), self.statemp
