load("miniconda3")
execute {cmd="source activate   /hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/Python/py2_pytorch_020", modeA={"load"}}
execute {cmd="conda deactivate", modeA={"unload"}}
setenv("DL_HOME", "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/dl-gpu-utils")
prepend_path("PATH", "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/dl-gpu-utils/scripts")
