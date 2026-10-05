"""
Download EVE level 2B line files and reduce them to the hourly cache used by the search.

    python scripts/build_eve_cache.py --start 2010-04-20 --end 2014-05-26

Resumable; rerun to retry failed days. ``--prune`` deletes each raw L2B file once reduced.
"""

import argparse
import logging

from evefip import config
from evefip.search import eve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2010-04-20")
    p.add_argument("--end", default=str(config.MEGSA_END.date()))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--prune", action="store_true", help="delete raw L2B files after reduction")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    eve.build_hourly_cache(args.start, args.end, workers=args.workers, prune=args.prune)


if __name__ == "__main__":
    main()
