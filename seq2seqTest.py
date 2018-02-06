from seq2seq import Seq2Seq
import pickle
import numpy as np
import json
import unittest


class seq2seqTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass
        # execute before every test case

    def tearDown(self):
        pass
        # execute after every test case

    def test01_readIntentTestJson(self):
        [input_characters, target_characters, input_character_index, target_character_index, input_bits, output_bits,
         max_encoder_seq_length, max_decoder_seq_length] = pickle.load(open('parameters.pkl', 'rb'))
        s2s = Seq2Seq(max_decoder_seq_length, target_character_index['\t'], input_bits=input_bits,
                      output_bits=output_bits)

        s2s.loadWeights()

        readFile = open('Dataset/gaplant/data.json', encoding='utf-8')
        for idx, line in enumerate(readFile):
            jsonLine = json.loads(line)

            sourceText = str.strip(jsonLine["source"])
            targetText = str.strip(jsonLine["target"])

            encoder_input_data = np.zeros((1, max_encoder_seq_length, input_bits), dtype='float32')
            for t, char in enumerate(sourceText):
                encoder_input_data[0, t, input_character_index[char]] = 1.
            s2s.setInput(encoder_input_data)
            outSentense = ''

            out = s2s.predictNext();
            out_index = np.argmax(out)
            outChar = target_characters[out_index]
            outSentense += outChar
            while str(outChar) != str('\n') and len(outSentense) < max_decoder_seq_length:
                out = s2s.predictNext();
                out_index = np.argmax(out)
                outChar = target_characters[out_index]
                outSentense += outChar

            outSentense = str.strip(outSentense)

            print(str(idx) + "번 확인! ", "source[", sourceText, "]", "expected[", targetText, "]", "outputText[", targetText, "]")
            self.assertEqual(str.strip(outSentense), str.strip(targetText), str(idx) + "번 확인!")


        readFile.close()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(seq2seqTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
