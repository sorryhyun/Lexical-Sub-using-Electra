from tester import tester

if __name__ == '__main__':
    tester(test_type='w2v', preprocessed_corpus_path='./data/Modu.json').test()

    tester(test_type='bert', preprocessed_corpus_path='./data/Modu.json').test()

    tester(test_type='bert-ls', preprocessed_corpus_path='./data/Modu.json').test()

    tester(test_type='electra', preprocessed_corpus_path='./data/Modu.json').test()

    tester(test_type='electra-ls', preprocessed_corpus_path='./data/Modu.json').test()
