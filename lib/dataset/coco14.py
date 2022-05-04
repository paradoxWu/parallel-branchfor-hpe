import sys
sys.path.append('.')
from lib.dataset import Dataset2D
from lib.core.config import COCO14_DIR
class Coco14(Dataset2D):
    def __init__(self, seqlen, overlap=0., debug=False): #原来overlap=0.75
        db_name = 'coco'

        super(Coco14, self).__init__(
            seqlen = seqlen,
            folder=COCO14_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')

if __name__ == '__main__':
    
    data = Coco14(16)
    data.__getitem__(1)