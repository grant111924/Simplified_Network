import argparse
from model.simpifiedNet import GoodNet2 as GoodNet2

class runner():
    def __init__(self):
        pass 
    def data():
        pass
    def train():
        pass
    def test():
        pass
    def predit():
        pass
def main():
    parser = argparse.ArgumentParser(prog='main',description='Process some integers.')

    parser.add_argument('-d','--dir',help='input data dir address include test and train folder')

File structure
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    args = parser.parse_args()
    print(args)
    # print(args.accumulate(args.integers))
    # GoodNet2(400)
    pass 

if __name__ == "__main__":
    # x = torch.rand(5, 3)
    # print(x)
    main();
    pass

