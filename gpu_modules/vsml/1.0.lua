load("python-genchem")

VSML_DIR = "/gpfs/workspace/users/sreshv/Projects/AI_Library_Design/VSML/vsml"

NUMEXPR_MAX_THREADS = "8"
setenv("NUMEXPR_MAX_THREADS", "8")
setenv("SINGULARITYENV_NUMEXPR_MAX_THREADS", "8")

VSML_BIN_DIR = pathJoin(VSML_DIR, "bin")
CHEMPROP_DIR = pathJoin(VSML_DIR, "chemprop_repo")
PQSAR_DIR = pathJoin(VSML_DIR, "pqsar_repo")
prepend_path("PYTHONPATH", VSML_DIR)
prepend_path("PYTHONPATH", CHEMPROP_DIR)
prepend_path("PYTHONPATH", PQSAR_DIR)
setenv("SINGULARITYENV_PYTHONPATH", pathJoin(os.getenv("PYTHONPATH"),""))
 
prepend_path("PATH", VSML_BIN_DIR)
setenv("SINGULARITYENV_PREPEND_PATH", VSML_BIN_DIR)

