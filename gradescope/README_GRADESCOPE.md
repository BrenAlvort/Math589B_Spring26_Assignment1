# Gradescope autograder packaging (staff)

Gradescope expects an autograder zip containing a `run_autograder` executable at the top level.

Suggested workflow:
1. Copy the contents of `gradescope/source/` into your Gradescope autograder zip root.
2. In Gradescope, set the programming language environment to one that has Python + build tools.
3. The script assumes the student's submission is a full repo with `csrc/build.sh`.

You will want to replace the placeholder scoring thresholds in `run_autograder`
with values calibrated against a staff reference solution.
