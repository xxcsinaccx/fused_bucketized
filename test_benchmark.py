import subprocess
commands = [
    "make",
    "./applications_fused_bucketized",
]
workdir = "/fused_bucketized"
for cmd in commands:
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=workdir)
    if result.returncode != 0:
        print(f"fail: {cmd}")
        break
