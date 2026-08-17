# RISC-V CPU Implementation (Single-Cycle, Extended, and Pipelined)

A from-scratch simulation of a RISC-V CPU in Python, built in three stages: a single-cycle
datapath, an extension adding JAL/JALR jump-and-link instructions, and a 5-stage pipelined
version with hazard detection.

## Overview

We modeled the classic RISC-V datapath — Fetch, Decode, Execute, Memory, Writeback — as
discrete Python functions operating on a global PC, a 32-entry register file, and
word-addressed data memory. This is a software model of the hardware stages, not
cycle-accurate RTL — the goal was to work through how control signals, immediate encoding,
and pipeline hazards actually behave.

## Part 1: Single-Cycle CPU

We implemented the base RV32I datapath: `Fetch()`, `Decode()`, `Execute()`, `Mem()`,
`Writeback()`, plus a `ControlUnit()` that derives control signals from the opcode and an ALU
control resolver that finalizes the operation once funct3/funct7 are known. Supports AND, OR,
ADD, SUB, load/store, and branch-on-equal.

The trickiest part of this stage was the branch (SB-type) immediate — RISC-V splits it across
non-contiguous bit positions in the instruction word, so decode has to reassemble it correctly
before sign-extension.

## Part 2: JAL / JALR Extension

We added unconditional jump-and-link support: `JAL` (PC-relative) and `JALR`
(register-relative), including saving the return address (PC+4) to the destination register.
JAL's 20-bit immediate is scattered across four non-contiguous bit ranges in the instruction
encoding, which made correct reassembly in `Decode()` the main challenge — along with the
RISC-V-mandated requirement that JALR clear the low bit of its computed target address.

## Part 3: Pipelined CPU

We extended the design into a 5-stage pipeline (IF/ID/EX/MEM/WB) using Python dataclasses as
pipeline registers, with all stages reading prior-cycle state and writing to "next" register
buffers to correctly model synchronous hardware updates. We implemented hazard detection that
stalls fetch (injecting a bubble) whenever an in-flight instruction's destination register
matches an incoming instruction's source register — data forwarding is not implemented, so
RAW hazards cost real pipeline bubbles rather than being resolved for free.

## Running it
```
python cpu.py

# enter a program file when prompted, e.g. sample_part1.txt / sample_part2.txt
```

Single-cycle mode prints register/memory changes and PC after each instruction. Pipelined
mode prints output when each instruction reaches WB — so the first instruction reports
completion at cycle 5, reflecting the pipeline depth.

## Known limitations

- No data forwarding — dependent instructions close together incur multiple stall cycles
- No branch prediction — branches resolve in EX and flush IF/ID, ID/EX on taken branches
- Models a functional subset of RV32I (no full ISA coverage)
