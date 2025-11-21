import subprocess
commands = "./applications_fused_bucketized"
workdir = "/fused_bucketized"
result = subprocess.run(commands, shell=True, cwd=workdir)
