from konlpy.tag import Okt
ko = Okt()

def tokenize(sentence):
    result = ''
    reslist = ko.pos(sentence, norm=True, stem=True)
    reslist = [tempword[0] for tempword in reslist
               if (tempword[1][0] != 'J' and tempword[1][0] != 'E'
                   and tempword[1][0] != 'P')]

    if reslist:  # 만약 이번에 읽은 데이터에 명사가 존재할 경우에만
        result = reslist # 결과에 저장

    return result

def tokenize_corpus(fin_path, fout_path):
    import json, os
    if os.path.isfile(fout_path):
        print("skipping preprocessing... (preprocessing successed?)")
        return

    with open(fin_path, 'r') as f:
        lines = json.load(f)

    result = []
    import multiprocessing
    with multiprocessing.Pool(8) as pool:
        result = pool.map(tokenize, lines)
    pool.close()
    pool.join()

    with open(fout_path, 'w', encoding="UTF-8") as f:
        json.dump(result, f)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('arguments error!')
    else:
        tokenize_corpus(fin_path=sys.argv[1], fout_path=sys.argv[2])
