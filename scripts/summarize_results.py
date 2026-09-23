from qbench.cli import main


if __name__ == "__main__":
    import sys
    raise SystemExit(main(["summarize", *sys.argv[1:]]))
