load("medsci")
load("met-conda")
load("cuda/10.2.89-gcc-9.2.0-ev36")
execute {cmd="source activate /gpfs/workspace/projects/methods_development/deep_learning/conda/envs/genchem", modeA={"load"}}
execute {cmd="source deactivate", modeA={"unload"}}
setenv("DL_HOME", "/gpfs/workspace/projects/methods_development/deep_learning/dl-gpu-utils")
prepend_path("PATH", "/gpfs/workspace/projects/methods_development/deep_learning/dl-gpu-utils/scripts")

SING_HOME="/gpfs/workspace/projects/methods_development/deep_learning/dl-gpu-utils/singularity"
setenv("SING_HOME", SING_HOME)
setenv("SINGULARITY_BINDPATH", "/gpfs,/delta")
setenv("SINGULARITY_CONTAINER", SING_HOME.."/genchem2/genchem.sif")
set_alias("SIPYTHON", "singularity --nocolor run --nv $SINGULARITY_CONTAINER python")
set_alias("SIRUN", "singularity --nocolor run --nv $SINGULARITY_CONTAINER")
