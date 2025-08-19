import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trader.core.train_buy_model import train as train_buy_model
from trader.core.train_sell_model import train as train_sell_model

def main(input_file):
    train_buy_model(input_file)
    train_sell_model(input_file)

if __name__ == "__main__":
    main(sys.argv[1])
