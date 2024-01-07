--conflict("py3gpu")
load("cuda-cudnn/9-705")
load("miniconda3")
execute {cmd="source activate /hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/Python/py3_pytorch_041_cudatk92", modeA={"load"}}
execute {cmd="conda deactivate", modeA={"unload"}}
setenv("DL_HOME", "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/dl-gpu-utils")
prepend_path("PATH", "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/dl-gpu-utils/scripts")
