from Bio.Seq import Seq
from Bio import SeqIO
import os, sys, site, shutil
import subprocess
import argparse


def reverse_complement(fastain,revfasta):
    out = open(revfasta,'w')
    for seq_record in SeqIO.parse(fastain,"fasta"):
        rev_seq = seq_record.reverse_complement()
        SeqIO.write(rev_seq,out,"fasta")
    out.close()




def format(seqin, width=60):
    """ Make human readable fasta format """
    seq = []
    j = 0
    for i in range(width, len(seqin),width):
        seq.append(seqin[j:i])
        j = i
    seq.append(seqin[j:])
    return '\n'.join(seq)



class Indexer():
    def __init__(self,fasta,fastaidx,window=100):
        self.fasta = fasta       #Input fasta
        self.fastaidx = fastaidx #Output fasta index
        self.faidx = {} #internal dict for storing byte offset values
        self.window=window
    """
    Develop an index similar to samtools faidx
    speciesName: give the option to pick out only the accession id
    """
    def index(self):
        idx = []
        #name = name of sequence
        #seqLen = length of sequence without newline characters
        #lineLen = number of characters per line
        #byteLen = length of sequence, including newline characters
        #myByteoff = byte offset of sequence
        name, seqLen, byteoff, myByteoff, lineLen, byteLen = None, 0, 0, 0, 0, 0
        index_out = open(self.fastaidx,'w')

        with open(self.fasta,'r') as handle:
            for ln in handle:
                lnlen = len(ln)

                if len(ln)==0: break
                if ln[0]==">":
                    if name is not None:
                        #Handle stupid parsing scenario
                        if name[:3]=="gi|":acc = name.split('|')[3]
                        else: acc = name
                        index_out.write('\t'.join(map(str, [acc, seqLen, myByteoff,
                                                            lineLen, byteLen])))
                        index_out.write('\n')
                        seqLen = 0
                    myByteoff = byteoff + lnlen
                    seqLen = 0
                    if ' ' in ln:
                        name = ln[1:ln.index(' ')].rstrip()
                    else:
                        name = ln[1:].rstrip()
                    byteoff+=lnlen
                else:

                    byteLen = max(byteLen, len(ln))
                    ln = ln.rstrip()
                    lineLen = max(lineLen, len(ln))
                    seqLen += len(ln)
                    byteoff += lnlen
        if name is not None:
            if name[:3]=="gi|": acc = name.split('|')[3]
            else: acc = name
            index_out.write('\t'.join(map(str, [acc, seqLen, myByteoff,
                                                lineLen, byteLen])))
            index_out.write('\n')
        index_out.close()

    """ Load fasta index """
    def load(self):
        with open(self.fastaidx,'r') as handle:
            for line in handle:
                line=line.strip()
                cols=line.split('\t')

                chrom = cols[0]
                seqLen,byteOffset,lineLen,byteLen = map(int,cols[1:])
                self.faidx[chrom]=(seqLen,byteOffset,lineLen,byteLen)

    def __getitem__(self,defn):

        seqLen,byteOffset,lineLen,byteLen=self.faidx[defn]
        return self.fetch(defn,1,seqLen)

    def fetch(self, defn, start, end):
        """ Retrieve a sequence based on fasta index """
        if len(self.faidx)==0:
            print("Empty table ...")
        assert type(1)==type(start)
        assert type(1)==type(end)
        self.fasta_handle = open(self.fasta,'r')
        seq=""
        if defn not in self.faidx:
            raise ValueError('Sequence %s not found in reference' % defn)
        seqLen,byteOffset,lineLen,byteLen=self.faidx[defn]
        start = start-1
        pos = byteOffset+start/lineLen*byteLen+start%lineLen
        self.fasta_handle.seek(pos)
        while len(seq)<end-start:
            line=self.fasta_handle.readline()
            line=line.rstrip()
            seq=seq+line
        self.fasta_handle.close()
        return seq[:end-start]
