#!/bin/bash -l
# ****************************************************************************
# Simple script to automate the unzipping of Rosetta silent files. 
# Updated: 24 March 2018
#
# NOTE:
#   This script's purpose is to allow you to set a large number of silent files
#   to be copied and unzipped at once, and the case that it illustrates
#   provides a good example of why it can be useful - in this case it would
#   take over 45 minutes to manually do all of this. 
#
#   Copying these silent files manually is annoying but not too long, but 
#   un-bzipping them is a very tedious process and doing them one-by-one is 
#   miserable. Set this script up to match your cases (it will take a few minutes, 
#   but will be worth it) and then let it run in the background. 
#
#   Obviously there is NO ERROR CHECKING HERE!
# ****************************************************************************
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_01_cm/rb_05_30_65910_109981__t000__0_C1_SAVE_ALL_OUT__376932_0.all.out.bz2 ./01-01_0c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_01_cm/rb_05_30_65910_109981__t000__1_C1_SAVE_ALL_OUT__376932_0.all.out.bz2 ./01-01_1c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_01_cm/rb_05_30_65910_109981__t000__2_C1_SAVE_ALL_OUT__376932_0.all.out.bz2 ./01-01_2c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_01_cm/rb_05_30_65910_109981__t000__3_C1_SAVE_ALL_OUT__376932_0.all.out.bz2 ./01-01_3c1 &

cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_02_cm/rb_05_30_65910_109982__t000__0_C1_SAVE_ALL_OUT__376940_0.all.out.bz2 ./01-02_0c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_02_cm/rb_05_30_65910_109982__t000__1_C1_SAVE_ALL_OUT__376940_0.all.out.bz2 ./01-02_1c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_02_cm/rb_05_30_65910_109982__t000__2_C1_SAVE_ALL_OUT__376940_0.all.out.bz2 ./01-02_2c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_02_cm/rb_05_30_65910_109982__t000__3_C1_SAVE_ALL_OUT__376940_0.all.out.bz2 ./01-02_3c1 &
cp /global/project/projectdirs/m1532/downloads/CASP12/T0898/01_02_cm/rb_05_30_65910_109982__t000__4_C1_SAVE_ALL_OUT__376940_0.all.out.bz2 ./01-02_4c1 &
wait

echo 'Copying complete...beginning unzip'

bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-01_0c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-01_1c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-01_2c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-01_3c1/*.bz2 &

bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-02_0c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-02_1c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-02_2c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-02_3c1/*.bz2 &
bzip2 -dk /global/cscratch1/sd/tjosc/casp_data/t0898-d1/01-02_4c1/*.bz2 &
wait

echo 'Unzipping complete.'

echo 'Done moving and unzipping .bz2 files.'
