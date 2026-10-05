"""Thin spark-submit entry point; install the same course package on every node."""

from mu_course.run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
