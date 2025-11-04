#!/usr/bin/env python3
import time

def main():
    t0 = time.perf_counter()
    # put your benchmark code here later
    time.sleep(0.1)
    t1 = time.perf_counter()
    print(f"OK. Elapsed: {(t1 - t0)*1000:.2f} ms")

if __name__ == "__main__":
    main()
