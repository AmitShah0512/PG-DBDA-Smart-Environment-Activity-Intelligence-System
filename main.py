# main.py
import sys

def usage():
    print("python main.py build|train|realtime|export")

def main():
    if len(sys.argv) < 2:
        usage()
        return

    m = sys.argv[1].lower()

    if m == "build":
        from core.dataset import build_dataset
        build_dataset()

    elif m == "train":
        from core.train import train_model
        train_model()

    elif m == "realtime":
        from core.realtime import run_realtime
        run_realtime()

    elif m == "export":
        from analytics.export import export_actions
        export_actions()

    else:
        usage()

if __name__ == "__main__":
    main()
