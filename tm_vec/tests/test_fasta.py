import os
import unittest
from tm_vec.fasta import Indexer


class TestIndex(unittest.TestCase):
    def setUp(self):
        entries = ['>testseq10_1',
                   'AGCTACT',
                   '>testseq10_2',
                   'AGCTAGCT',
                   '>testseq40_2',
                   'AAGCTAGCT',
                   '>testseq40_5',
                   'AAGCTAGCT\n'*100
                   ]
        self.fasta = "test.fa"
        self.fastaidx = "test.fai"
        self.revfasta = "rev.fa"
        open(self.fasta,'w').write('\n'.join(entries))
    def tearDown(self):
        if os.path.exists(self.fasta):
            os.remove(self.fasta)
        if os.path.exists(self.fastaidx):
            os.remove(self.fastaidx)
        if os.path.exists(self.revfasta):
            os.remove(self.revfasta)
    def testIndex(self):
        indexer = Indexer(self.fasta,self.fastaidx)
        indexer.index()
        indexer.load()

        seq = indexer.fetch("testseq10_1",1,4)
        self.assertEquals("AGCT",seq)
        seq = indexer.fetch("testseq40_5",1,13)
        self.assertEquals("AAGCTAGCTAAGC",seq)
        seq = indexer.fetch("testseq40_5",1,900)
        self.assertEquals("AAGCTAGCT"*100,seq)
    def testGet(self):
        indexer = Indexer(self.fasta,self.fastaidx)
        indexer.index()
        indexer.load()
        self.assertEquals('AGCTACT',indexer["testseq10_1"])
        self.assertEquals('AGCTAGCT',indexer["testseq10_2"])
        self.assertEquals('AAGCTAGCT',indexer["testseq40_2"])
        self.assertEquals('AAGCTAGCT'*100,indexer["testseq40_5"])

class TestFasta(unittest.TestCase):
    def setUp(self):
        entries = ['>testseq1',
                   'AGCTACT',
                   '>testseq2',
                   'AGCTAGCT',
                   '>testseq2',
                   'AAGCTAGCT'
                   '>testseq3',
                   'AAGCTAGCT\n'*100
                   ]
        self.fasta = "test.fa"
        self.fastaidx = "test.fai"
        open(self.fasta,'w').write('\n'.join(entries))

    def tearDown(self):
        if os.path.exists(self.fasta):
            os.remove(self.fasta)
        if os.path.exists(self.fastaidx):
            os.remove(self.fastaidx)


if __name__ == "__main__":
    unittest.main()
