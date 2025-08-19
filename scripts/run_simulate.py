import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trader.core.simulate import simulate

def main(input_file):
    simulate(input_file)

if __name__ == "__main__":
    main(sys.argv[1])
