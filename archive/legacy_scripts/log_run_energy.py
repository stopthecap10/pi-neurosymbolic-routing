import argparse, csv, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

def now_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id","system","tier","csv_used","model_file","repeats",
            "start_mWh","end_mWh",
            "elapsed_s","exit_code",
            "out_csv","trials_csv"
        ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)   # phi2_q8_grammar, phi2_q8_nogrammar, hybrid_v1, hybrid_v2, tinyllama, sympy_only
    ap.add_argument("--tier", required=True)     # T1, T2, T3
    ap.add_argument("--csv_used", required=True)
    ap.add_argument("--model_file", required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--log_csv", default="outputs/energy_log.csv")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)  # command after --
    args = ap.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        print("ERROR: you must pass the command after a literal --")
        print("Example: python3 src/log_run_energy.py ... -- python3 src/run_phi2_server_runner.py ...")
        sys.exit(2)
    cmd = args.cmd[1:]

    run_id = now_id()
    log_path = Path(args.log_csv)
    ensure_header(log_path)

    print(f"RUN {run_id} system={args.system} tier={args.tier}")
    print("ENTER start_mWh from watt meter (just the number): ", end="", flush=True)
    start_mWh = input().strip()

    t0 = time.time()
    try:
        p = subprocess.run(cmd)
        exit_code = int(p.returncode)
    except KeyboardInterrupt:
        exit_code = 130
    elapsed_s = time.time() - t0

    print("ENTER end_mWh from watt meter (just the number): ", end="", flush=True)
    end_mWh = input().strip()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            run_id, args.system, args.tier, args.csv_used, args.model_file, args.repeats,
            start_mWh, end_mWh,
            f"{elapsed_s:.3f}", exit_code,
            args.out_csv, args.trials_csv
        ])

    print(f"Logged to {log_path} (elapsed_s={elapsed_s:.3f}, exit_code={exit_code})")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
