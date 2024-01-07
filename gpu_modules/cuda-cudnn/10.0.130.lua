help([[
Load CUDA 10.0.130/CuDNN-7.6

]])
whatis("Version: 10.0.130/7.6")
whatis("Keywords: NVIDIA, GPU, CUDA")
whatis("URL: https://github.com/PfizerRD/dl-gpu-utils/wiki")
whatis("Description: Load Nvidida CUDA deep learning libraries")

setenv(       "CUDA_HOME",           "/nfs/grid/software/RHEL7/medsci/software/spack/rhel7-p100/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/cuda-10.0.130-y22r6scgqudbo4chffvdvwuuoxz5x56m")
prepend_path( "PATH",                "/nfs/grid/software/RHEL7/medsci/software/spack/rhel7-p100/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/cuda-10.0.130-y22r6scgqudbo4chffvdvwuuoxz5x56m/bin")
prepend_path( "LD_LIBRARY_PATH",     "/nfs/grid/software/RHEL7/medsci/software/spack/rhel7-p100/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/cuda-10.0.130-y22r6scgqudbo4chffvdvwuuoxz5x56m/lib64")
--prepend_path( "LD_LIBRARY_PATH",     "/nfs/grid/software/RHEL7/medsci/software/spack/rhel7-p100/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/cudnn-7.6-sktofmofm2outslgtx3wmpj7jopzdpta/lib64")
prepend_path( "LIBRARY_PATH",        "/nfs/grid/software/RHEL7/medsci/software/spack/rhel7-p100/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/cuda-10.0.130-y22r6scgqudbo4chffvdvwuuoxz5x56m/lib64")
append_path(  "CPATH",               "/nfs/grid/software/RHEL7/medsci/software/spack/rhel7-p100/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/cuda-10.0.130-y22r6scgqudbo4chffvdvwuuoxz5x56m/include")
setenv(       "MKL_THREADING_LAYER", "GNU")
