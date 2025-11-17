# cli.py
"""
Thin wrapper: forward all CLI args to trainer.task.main()
"""

from trainer.task import main

if __name__ == "__main__":
    # trainer.task.main() 自己会用 argparse 解析 sys.argv
    main()
