from dataclasses import dataclass
from typing import List


# =========================================================
# Bit utilities
# =========================================================

def slice_bits(instr: str, hi: int, lo: int) -> str:
    start = 31 - hi
    end = 31 - lo
    return instr[start:end + 1]


def bits_as_int(b: str) -> int:
    return int(b, 2)


def sign_extend(value: int, bit_width: int) -> int:
    sign_bit = 1 << (bit_width - 1)
    mask = (1 << bit_width) - 1
    value &= mask
    return (value ^ sign_bit) - sign_bit


# =========================================================
# Opcode constants
# =========================================================

OP_R      = "0110011"
OP_IMM    = "0010011"
OP_LOAD   = "0000011"
OP_STORE  = "0100011"
OP_BRANCH = "1100011"

REG_NAMES = [
    "zero","ra","sp","gp","tp","t0","t1","t2",
    "s0","s1","a0","a1","a2","a3","a4","a5",
    "a6","a7","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","t3","t4","t5","t6"
]


# =========================================================
# Pipeline register structures
# =========================================================

@dataclass
class IF_ID:
    valid: bool = False
    instr: str = ""
    pc: int = 0
    next_pc: int = 0


@dataclass
class ID_EX:
    valid: bool = False
    instr: str = ""
    pc: int = 0
    next_pc: int = 0

    opcode: str = ""
    funct3: str = ""
    funct7: str = ""

    rd_idx: int = 0
    rs1_idx: int = 0
    rs2_idx: int = 0

    rs1_val: int = 0
    rs2_val: int = 0
    imm_val: int = 0

    RegWrite: int = 0
    ALUSrc: int = 0
    MemRead: int = 0
    MemWrite: int = 0
    MemToReg: int = 0
    Branch: int = 0
    ALUOp: int = 0
    alu_ctrl: int = 0


@dataclass
class EX_MEM:
    valid: bool = False
    instr: str = ""
    pc: int = 0
    next_pc: int = 0

    rd_idx: int = 0
    rs2_val: int = 0

    alu_result: int = 0
    zero: int = 0
    branch_target: int = 0

    RegWrite: int = 0
    MemRead: int = 0
    MemWrite: int = 0
    MemToReg: int = 0
    Branch: int = 0


@dataclass
class MEM_WB:
    valid: bool = False
    instr: str = ""
    rd_idx: int = 0

    alu_result: int = 0
    mem_data: int = 0
    mem_msg: str = ""

    RegWrite: int = 0
    MemToReg: int = 0


# =========================================================
# Global machine state
# =========================================================

pc = 0
total_clock_cycles = 0

rf = [0] * 32
d_mem = [0] * 32
instructions: List[str] = []

if_id = IF_ID()
id_ex = ID_EX()
ex_mem = EX_MEM()
mem_wb = MEM_WB()

halt_fetch = False


# =========================================================
# Control logic helpers
# =========================================================

def control_unit(opcode: str):
    RegWrite = ALUSrc = MemRead = MemWrite = MemToReg = Branch = 0
    ALUOp = 0
    alu_ctrl = 0b0010  # default ADD

    if opcode == OP_R:
        RegWrite = 1
        ALUOp = 0b10

    elif opcode == OP_IMM:
        RegWrite = 1
        ALUSrc = 1
        ALUOp = 0b10

    elif opcode == OP_LOAD:
        RegWrite = 1
        ALUSrc = 1
        MemRead = 1
        MemToReg = 1
        ALUOp = 0b00
        alu_ctrl = 0b0010

    elif opcode == OP_STORE:
        ALUSrc = 1
        MemWrite = 1
        ALUOp = 0b00
        alu_ctrl = 0b0010

    elif opcode == OP_BRANCH:
        Branch = 1
        ALUOp = 0b01
        alu_ctrl = 0b0110

    return {
        "RegWrite": RegWrite,
        "ALUSrc": ALUSrc,
        "MemRead": MemRead,
        "MemWrite": MemWrite,
        "MemToReg": MemToReg,
        "Branch": Branch,
        "ALUOp": ALUOp,
        "alu_ctrl": alu_ctrl,
    }


def resolve_alu_ctrl(opcode_bits: str, funct3: str, funct7: str, aluop: int, current_ctrl: int):
    if aluop != 0b10:
        return current_ctrl

    if opcode_bits == OP_R:
        if funct3 == "000" and funct7 == "0000000":
            return 0b0010  # add
        elif funct3 == "000" and funct7 == "0100000":
            return 0b0110  # sub
        elif funct3 == "111":
            return 0b0000  # and
        elif funct3 == "110":
            return 0b0001  # or
    else:
        if funct3 == "000":
            return 0b0010  # addi
        elif funct3 == "111":
            return 0b0000  # andi
        elif funct3 == "110":
            return 0b0001  # ori

    return current_ctrl


def decode_immediate(instr: str, opcode_bits: str) -> int:
    if opcode_bits in (OP_IMM, OP_LOAD):
        raw = bits_as_int(slice_bits(instr, 31, 20))
        return sign_extend(raw, 12)

    elif opcode_bits == OP_STORE:
        hi = slice_bits(instr, 31, 25)
        lo = slice_bits(instr, 11, 7)
        raw = bits_as_int(hi + lo)
        return sign_extend(raw, 12)

    elif opcode_bits == OP_BRANCH:
        b31 = slice_bits(instr, 31, 31)
        b30_25 = slice_bits(instr, 30, 25)
        b11_8 = slice_bits(instr, 11, 8)
        b7 = slice_bits(instr, 7, 7)
        raw = bits_as_int(b31 + b7 + b30_25 + b11_8)
        return sign_extend(raw, 12)

    return 0


# =========================================================
# Hazard detection (no forwarding)
# =========================================================

def instruction_uses_rs2(opcode_bits: str) -> bool:
    return opcode_bits in (OP_R, OP_STORE, OP_BRANCH)


def should_stall_decode() -> bool:
    if not if_id.valid:
        return False

    instr = if_id.instr
    opcode_bits = slice_bits(instr, 6, 0)
    rs1_idx = bits_as_int(slice_bits(instr, 19, 15))
    rs2_idx = bits_as_int(slice_bits(instr, 24, 20))

    uses_rs2 = instruction_uses_rs2(opcode_bits)

    # No forwarding: if a source register is waiting to be written by a still-active older instruction,
    # stall decode/fetch and inject a bubble into EX.
    for older in (id_ex, ex_mem):
        if not older.valid:
            continue

        if not getattr(older, "RegWrite", 0):
            continue

        older_rd = getattr(older, "rd_idx", 0)
        if older_rd == 0:
            continue

        if rs1_idx == older_rd:
            return True
        if uses_rs2 and rs2_idx == older_rd:
            return True

    return False


# =========================================================
# Stage functions
# =========================================================

def writeback_stage():
    global rf

    if not mem_wb.valid:
        return None

    completed_msg = {
        "reg_msg": "",
        "mem_msg": mem_wb.mem_msg,
    }

    if mem_wb.RegWrite and mem_wb.rd_idx != 0:
        write_val = mem_wb.mem_data if mem_wb.MemToReg else mem_wb.alu_result
        rf[mem_wb.rd_idx] = write_val
        completed_msg["reg_msg"] = f"x{mem_wb.rd_idx} is modified to 0x{write_val & 0xFFFFFFFF:X}"

    return completed_msg


def memory_stage(next_mem_wb: MEM_WB):
    global d_mem

    if not ex_mem.valid:
        next_mem_wb.valid = False
        return

    mem_msg = ""
    mem_data = 0

    if ex_mem.MemRead:
        mem_index = ex_mem.alu_result // 4
        mem_data = d_mem[mem_index]

    elif ex_mem.MemWrite:
        mem_index = ex_mem.alu_result // 4
        d_mem[mem_index] = ex_mem.rs2_val
        mem_msg = f"memory 0x{ex_mem.alu_result:X} is modified to 0x{ex_mem.rs2_val:X}"

    next_mem_wb.valid = True
    next_mem_wb.instr = ex_mem.instr
    next_mem_wb.rd_idx = ex_mem.rd_idx
    next_mem_wb.alu_result = ex_mem.alu_result
    next_mem_wb.mem_data = mem_data
    next_mem_wb.mem_msg = mem_msg
    next_mem_wb.RegWrite = ex_mem.RegWrite
    next_mem_wb.MemToReg = ex_mem.MemToReg


def execute_stage(next_ex_mem: EX_MEM):
    global pc, if_id

    if not id_ex.valid:
        next_ex_mem.valid = False
        return False

    operand_a = id_ex.rs1_val
    operand_b = id_ex.imm_val if id_ex.ALUSrc else id_ex.rs2_val

    if id_ex.alu_ctrl == 0b0000:
        alu_result = operand_a & operand_b
    elif id_ex.alu_ctrl == 0b0001:
        alu_result = operand_a | operand_b
    elif id_ex.alu_ctrl == 0b0010:
        alu_result = operand_a + operand_b
    elif id_ex.alu_ctrl == 0b0110:
        alu_result = operand_a - operand_b
    else:
        alu_result = 0

    zero = 1 if alu_result == 0 else 0
    branch_target = id_ex.pc + (id_ex.imm_val << 1)

    next_ex_mem.valid = True
    next_ex_mem.instr = id_ex.instr
    next_ex_mem.pc = id_ex.pc
    next_ex_mem.next_pc = id_ex.next_pc
    next_ex_mem.rd_idx = id_ex.rd_idx
    next_ex_mem.rs2_val = id_ex.rs2_val
    next_ex_mem.alu_result = alu_result
    next_ex_mem.zero = zero
    next_ex_mem.branch_target = branch_target
    next_ex_mem.RegWrite = id_ex.RegWrite
    next_ex_mem.MemRead = id_ex.MemRead
    next_ex_mem.MemWrite = id_ex.MemWrite
    next_ex_mem.MemToReg = id_ex.MemToReg
    next_ex_mem.Branch = id_ex.Branch

    # No branch prediction: resolve in EX, then redirect PC and flush younger instructions if taken
    if id_ex.Branch and zero:
        pc = branch_target
        # if_id.valid = False
        return True

    return False


def decode_stage(next_id_ex: ID_EX):
    if not if_id.valid:
        next_id_ex.valid = False
        return False

    if should_stall_decode():
        # inject bubble into EX stage
        next_id_ex.valid = False
        return True

    instr = if_id.instr
    opcode_bits = slice_bits(instr, 6, 0)
    funct3 = slice_bits(instr, 14, 12)
    funct7 = slice_bits(instr, 31, 25)

    rd_idx = bits_as_int(slice_bits(instr, 11, 7))
    rs1_idx = bits_as_int(slice_bits(instr, 19, 15))
    rs2_idx = bits_as_int(slice_bits(instr, 24, 20))

    rs1_val = rf[rs1_idx]
    rs2_val = rf[rs2_idx]
    imm_val = decode_immediate(instr, opcode_bits)

    ctrl = control_unit(opcode_bits)
    ctrl["alu_ctrl"] = resolve_alu_ctrl(opcode_bits, funct3, funct7, ctrl["ALUOp"], ctrl["alu_ctrl"])

    next_id_ex.valid = True
    next_id_ex.instr = instr
    next_id_ex.pc = if_id.pc
    next_id_ex.next_pc = if_id.next_pc
    next_id_ex.opcode = opcode_bits
    next_id_ex.funct3 = funct3
    next_id_ex.funct7 = funct7
    next_id_ex.rd_idx = rd_idx
    next_id_ex.rs1_idx = rs1_idx
    next_id_ex.rs2_idx = rs2_idx
    next_id_ex.rs1_val = rs1_val
    next_id_ex.rs2_val = rs2_val
    next_id_ex.imm_val = imm_val
    next_id_ex.RegWrite = ctrl["RegWrite"]
    next_id_ex.ALUSrc = ctrl["ALUSrc"]
    next_id_ex.MemRead = ctrl["MemRead"]
    next_id_ex.MemWrite = ctrl["MemWrite"]
    next_id_ex.MemToReg = ctrl["MemToReg"]
    next_id_ex.Branch = ctrl["Branch"]
    next_id_ex.ALUOp = ctrl["ALUOp"]
    next_id_ex.alu_ctrl = ctrl["alu_ctrl"]

    return False


def fetch_stage(next_if_id: IF_ID, stall_fetch: bool):
    global pc, halt_fetch

    if stall_fetch:
        next_if_id.valid = if_id.valid
        next_if_id.instr = if_id.instr
        next_if_id.pc = if_id.pc
        next_if_id.next_pc = if_id.next_pc
        return

    instr_index = pc // 4
    if instr_index >= len(instructions):
        next_if_id.valid = False
        halt_fetch = True
        return

    next_if_id.valid = True
    next_if_id.instr = instructions[instr_index].strip()
    next_if_id.pc = pc
    next_if_id.next_pc = pc + 4

    pc = pc + 4


# =========================================================
# Main simulation loop
# =========================================================

def pipeline_empty() -> bool:
    return (
        not if_id.valid and
        not id_ex.valid and
        not ex_mem.valid and
        not mem_wb.valid and
        halt_fetch
    )


def main():
    global pc, total_clock_cycles, instructions, halt_fetch
    global if_id, id_ex, ex_mem, mem_wb

    filename = input("Enter the program file name to run:\n").strip()
    try:
        with open(filename, "r") as f:
            instructions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return

    # Part 1 initial state
    rf[:] = [0] * 32
    d_mem[:] = [0] * 32
    rf[1] = 0x20
    rf[2] = 0x5
    rf[10] = 0x70
    rf[11] = 0x4
    d_mem[0x70 // 4] = 0x5
    d_mem[0x74 // 4] = 0x10

    pc = 0
    total_clock_cycles = 0
    halt_fetch = False

    if_id = IF_ID()
    id_ex = ID_EX()
    ex_mem = EX_MEM()
    mem_wb = MEM_WB()

    while True:
        if pipeline_empty():
            break

        total_clock_cycles += 1

        next_if_id = IF_ID()
        next_id_ex = ID_EX()
        next_ex_mem = EX_MEM()
        next_mem_wb = MEM_WB()

        wb_msg = writeback_stage()
        memory_stage(next_mem_wb)
        branch_taken = execute_stage(next_ex_mem)
        stall_decode = decode_stage(next_id_ex)

        # If branch taken, flush fetch/decode path this cycle
        if branch_taken:
            next_if_id.valid = False
            next_id_ex.valid = False
        else:
            fetch_stage(next_if_id, stall_decode)

        # Commit pipeline registers at end of cycle
        if_id = next_if_id
        id_ex = next_id_ex
        ex_mem = next_ex_mem
        mem_wb = next_mem_wb

        # Print only when an instruction finishes WB
        if wb_msg is not None:
            if total_clock_cycles > 1:
                print()

            print(f"total_clock_cycles {total_clock_cycles} :")
            if wb_msg["reg_msg"]:
                print(wb_msg["reg_msg"])
            if wb_msg["mem_msg"]:
                print(wb_msg["mem_msg"])
            print(f"pc is modified to 0x{pc:X}")

    print(f"\nprogram terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")


if __name__ == "__main__":
    main()