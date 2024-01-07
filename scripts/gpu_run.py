#!/bin/env python
from __future__ import print_function
import os
import time
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser("Run Python script on GPUs")
parser.add_argument("script", help='Python script to run.')
parser.add_argument("--job-name", help="job name", dest='job_name')
parser.add_argument(
    "--queue-name", help="queue name", dest='queue_name', default='gpu')
parser.add_argument(
    "--queue-type", help="queue type ('gpup100', 'gpuk80', 'gpuv100'), default: gpup100",
    dest='queue_type', default='gpuk80')
parser.add_argument(
    "--node", help="specific node [without .pfizer.com extension]")
parser.add_argument("--num-gpus",
                    help="number of gpus, default: 1",
                    dest="num_gpus",
                    type=int,
                    default=1
                    )
parser.add_argument(
    "--number-of-cpus", help="number of nodes, default: 1", dest="num_cpus", default=1)
parser.add_argument(
    "--module-path",
    help="path to gpu dl modules",
    default='/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/dl-gpu-utils/gpu_modules')
parser.add_argument(
    "--gpu-module", help="GPU module to load, default: py3gpu", default='py3gpu')
parser.add_argument(
    "--dry-run", help="print script and exit", action='store_true')
parser.add_argument(
    "--pythonpath", help="things to add to pythonpath", default='')
parser.add_argument(
    "--conda-env", "-c", help="Custom conda environment path (must be on HPCC)", default='')   
parser.add_argument("script_args", nargs=argparse.REMAINDER)
args = parser.parse_args()

# Time stamp
time_stamp = time.time()

# Job name
if not args.job_name:
    args.job_name = os.path.basename(
        args.script).split('.')[0] + '_{:.0f}'.format(time_stamp)

if args.node:
    if 'adwgpu700' in args.node:
        args.queue_type = 'gpuv100'
    args.node = '#BSUB -m {}'.format(args.node)
else:
    args.node = ''

if args.num_gpus:
    args.num_gpus = '#BSUB -R "select[ngpus>0] rusage[ngpus_excl_p={}]"'.format(
        args.num_gpus)

# SLA class name
if args.queue_type == 'gpup100':
    sla_class_flag = '#BSUB -sla gpu_p100_class'
elif args.queue_type == 'gpuv100':
    sla_class_flag = '#BSUB -sla gpu_adwgpu700_class'
else:
    sla_class_flag = ''

if args.conda_env:
    conda_env_command = "conda activate {:s}".format(args.conda_env)
else:
    conda_env_command = ""
    
bsub_opts = {
    'script': args.script,
    'cwd': os.getcwd(),
    'job_name': args.job_name,
    'stdout_file': args.job_name + '.stdout',
    'stderr_file': args.job_name + '.stderr',
    'queue_name': args.queue_name,
    'queue_type': args.queue_type,
    'node_spec': args.node,
    'num_gpus': args.num_gpus,
    'num_cpus': args.num_cpus,
    'module_path': args.module_path,
    'gpu_module': args.gpu_module,
    'pythonpath': args.pythonpath,
    'script_args': ' '.join(args.script_args),
    'sla_class_flag': sla_class_flag,
    'conda_env_command': conda_env_command,
}

BSUB_SCRIPT = """
#BSUB -J {job_name}
#BSUB -q {queue_name}
#BSUB -o {stdout_file}
#BSUB -e {stderr_file}
#BSUB -R {queue_type}
#BSUB -n {num_cpus}
{num_gpus}
{node_spec}
{sla_class_flag}


#-----------------------------
env 1>&2
uname -a 1>&2
#-----------------------------

module use {module_path}
module load {gpu_module}
{conda_env_command}
cd {cwd}
export PYTHONPATH=$PYTHONPATH:{pythonpath}
python {script} {script_args}
""".format(**bsub_opts)

script_name = "run_script_{:.0f}.lsf".format(time_stamp)
with open(script_name, 'w') as script_file:
    script_file.write(BSUB_SCRIPT)
if args.dry_run:
    sys.exit(0)

print("Submitting job {}...".format(bsub_opts["job_name"]))
job = subprocess.Popen(
    "source /hpc/grid/lsfhpcprodchem/conf/profile.lsf; bsub < {}".format(
        script_name),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    shell=True, )

result = job.stdout.read()
if result == "":
    error = job.stderr.read()
    print("ERROR: %s" % error, file=sys.stderr)
else:
    print(result)
os.remove(script_name)
