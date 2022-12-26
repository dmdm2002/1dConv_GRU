class param(object):
    def __init__(self):
        self.ROOT = 'D:/[DB]/Dacon/DNA'
        self.OUTPUT = f'{self.ROOT}/backup/CNN_GRU/try_1'
        self.OUTPUT_CKP = f'{self.OUTPUT}/ckp'
        self.OUTPUT_LOSS = f'{self.OUTPUT}/log'
        self.CKP_LOAD = False
        self.classes = {'A': 0, 'B': 1, 'C': 2}

        # Train or Test
        self.EPOCH = 100
        self.BATCHSZ = 1
        self.LR = 1e-3