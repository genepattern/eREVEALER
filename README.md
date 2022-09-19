# REVEALER



### REVEALER2 (repeated evaluation of variables conditional entropy and redundancy 2) is a method for identifying groups of genomic alterations that together associate  with a functional activation, gene dependency or drug response profile. The combination of these alterations explains a larger fraction of samples displaying functional target activation or sensitivity than any individual alteration considered in isolation. REVEALER2 can be applied to a wide variety of problems and allows prior relevant background knowledge to be incorporated into the model. Compared to original REVEALER, REVEALER2.0 can work on much larger sample size with much higher speed.

### REVEALER2 is separated into two parts: REVEALER_preprocess and REVEALER. If you start with MAF file or GCT file that you want to have a further filtering, then you should run REVEALER_process and then use output as input for REVEALER. If you have ready-to-use GCT format matrix, then you can directly run REVEALER. Explanation and general usage about REVEALER_preprocess and REVEALER is provided below.

## REVEALER preprocess
