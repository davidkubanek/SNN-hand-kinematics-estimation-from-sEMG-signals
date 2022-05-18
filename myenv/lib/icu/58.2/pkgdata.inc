GENCCODE_ASSEMBLY_TYPE=-a gcc-darwin
SO=dylib
SOBJ=dylib
A=a
LIBPREFIX=lib
LIB_EXT_ORDER=.58.2.dylib
COMPILE=/opt/concourse/worker/volumes/live/55ac5f65-0d2f-4c2c-41aa-7b10b5ab5750/volume/icu_1588025713469/_build_env/bin/clang -DU_ATTRIBUTE_DEPRECATED= -D_FORTIFY_SOURCE=2 -mmacosx-version-min=10.9 -I/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/include   -DU_HAVE_ATOMIC=1 -DU_HAVE_STRTOD_L=1  -march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -I/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/include -fdebug-prefix-map=/opt/concourse/worker/volumes/live/55ac5f65-0d2f-4c2c-41aa-7b10b5ab5750/volume/icu_1588025713469/work=/usr/local/src/conda/icu-58.2 -fdebug-prefix-map=/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv=/usr/local/src/conda-prefix -std=c99 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -fno-common -c
LIBFLAGS=-I/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/include  -dynamic
GENLIB=/opt/concourse/worker/volumes/live/55ac5f65-0d2f-4c2c-41aa-7b10b5ab5750/volume/icu_1588025713469/_build_env/bin/clang -dynamiclib -dynamic -march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -I/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/include -fdebug-prefix-map=/opt/concourse/worker/volumes/live/55ac5f65-0d2f-4c2c-41aa-7b10b5ab5750/volume/icu_1588025713469/work=/usr/local/src/conda/icu-58.2 -fdebug-prefix-map=/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv=/usr/local/src/conda-prefix -std=c99 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -Wl,-pie -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/lib -L/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/lib -Wl,-rpath,/Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/lib 
LDICUDTFLAGS=
LD_SONAME=-Wl,-compatibility_version -Wl,58 -Wl,-current_version -Wl,58.2 -install_name /Users/david/Documents/Code/SNN-hand-kinematics-estimation-from-sEMG-signals/myenv/lib/
RPATH_FLAGS=
BIR_LDFLAGS=
AR=/opt/concourse/worker/volumes/live/55ac5f65-0d2f-4c2c-41aa-7b10b5ab5750/volume/icu_1588025713469/_build_env/bin/x86_64-apple-darwin13.4.0-ar
ARFLAGS=r -c
RANLIB=/opt/concourse/worker/volumes/live/55ac5f65-0d2f-4c2c-41aa-7b10b5ab5750/volume/icu_1588025713469/_build_env/bin/x86_64-apple-darwin13.4.0-ranlib
INSTALL_CMD=/usr/bin/install -c
