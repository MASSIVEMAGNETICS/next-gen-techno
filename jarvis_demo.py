"""
J.A.R.V.I.S. Interactive Demo
Run this script to start JARVIS in always-on mode:

    python jarvis_demo.py
    python jarvis_demo.py --owner "Mr Stark"
"""

import argparse
from omega_tensor import JARVIS


def main():
    parser = argparse.ArgumentParser(
        description="Start JARVIS – Just A Rather Very Intelligent System"
    )
    parser.add_argument(
        "--owner",
        default="Sir",
        help="Name JARVIS uses when addressing you (default: 'Sir')",
    )
    args = parser.parse_args()

    j = JARVIS(owner=args.owner)
    j.run()


if __name__ == "__main__":
    main()
