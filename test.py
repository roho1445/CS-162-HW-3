import unittest
import pos_tagger

class TestLanguageModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.train_data, self.dev_data = pos_tagger.load_data()
        self.pos_tagger = pos_tagger.POSTagger(self.train_data)

    def test_tran_prob(self):
        self.assertAlmostEqual(self.pos_tagger.tran_prob[('FW-NN', 'NNS')], 0.04)
        self.assertAlmostEqual(self.pos_tagger.tran_prob[('VBG', 'JJT')], 0.0003351206434316354)

    def test_emis_prob(self):
        self.assertAlmostEqual(self.pos_tagger.emis_prob[('FW-NN', 'wei')], 0.04)
        self.assertAlmostEqual(self.pos_tagger.emis_prob[('NP', 'Lo')], 0.0002690824289173917)

    def test_init_prob(self):
        self.assertAlmostEqual(self.pos_tagger.init_prob('NP'), 0.0893)
        self.assertAlmostEqual(self.pos_tagger.init_prob('NN'), 0.0247)

    def test_viterbi(self):
        self.assertEqual(self.pos_tagger.viterbi(['Eddie', 'shouted', '.']), ['NP', 'VBD', '.'])
        self.assertEqual(self.pos_tagger.viterbi(['Mike', 'caught', 'the', 'ball', 'just', 'as', 'the', 'catcher', 'slid', 'into', 'the', 'bag', '.']), ['NP', 'VBD', 'AT', 'NN', 'RB', 'CS', 'AT', 'NN', 'VBD', 'IN', 'AT', 'NN', '.'])
        

if __name__ == '__main__':
    unittest.main()
