help([[
Load CUDA 9.0 / cuDNN 7.0.5 

]])
whatis("Version: 9.0/7.0.5")
whatis("Keywords: NVIDIA, GPU, CUDA")
whatis("URL: https://github.com/PfizerRD/dl-gpu-utils/wiki")
whatis("Description: Load Nvidida CUDA deep learning libraries")

setenv(       "CUDA_HOME",           "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/CUDA/cuda-9.0")
prepend_path( "PATH",                "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/CUDA/cuda-9.0/bin")
prepend_path( "LD_LIBRARY_PATH",     "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/CUDA/cuda-9.0/lib64")
prepend_path( "LIBRARY_PATH",        "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/CUDA/cuda-9.0/lib64")
append_path(  "CPATH",               "/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/CUDA/cuda-9.0/include")
setenv(       "MKL_THREADING_LAYER", "GNU")
