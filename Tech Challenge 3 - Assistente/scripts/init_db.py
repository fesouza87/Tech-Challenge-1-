from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from assistant.db import connect, ensure_schema, seed_synthetic


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db_path", default=os.path.join(os.path.dirname(__file__), "..", "data", "patients.db"))
    args = p.parse_args()

    db_path = os.path.abspath(args.db_path)
    conn = connect(db_path)
    ensure_schema(conn)
    seed_synthetic(conn)
    conn.close()
    print(f"Banco pronto: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
