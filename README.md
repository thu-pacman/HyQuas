# QCSimulator

## Compile and Run
1. Get the source code
```bash
git clone *****/QCSimulator.git --recursive
```

2. Compile the tensor transpose library `cutt`

modify `GENCODE_FLAGS` in `third-party/cutt/Makefile` for the target compute capability.

dependency: cuda >= 9.0, gcc < 8

```bash
cd third-party/cutt
make -j
```

3. Run a single circuit

```bash
cd scripts
./run.sh
```

4. Run all circuits and check the correctness
```bash
cd scripts
./check.sh
```
