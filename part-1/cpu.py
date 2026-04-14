# cpu.py
# CSE 140 Project – Part 1: Single-Cycle RISC-V CPU Simulator
# Supports: lw, sw, add, addi, sub, and, andi, or, ori, beq
#
# Reuses decode logic from HW3 (JoeSamuelRosh_NoahElliott.py)
 
 
# ─────────────────────────────────────────────
# Bit utilities (from HW3)
# ─────────────────────────────────────────────
 
def slice_bits(instr: str, hi: int, lo: int) -> str:
    """Return bits [hi:lo] inclusive. Bit 31 = instr[0], bit 0 = instr[31]."""
    start = 31 - hi
    end   = 31 - lo
    return instr[start:end + 1]
 
def bits_as_int(b: str) -> int:
    """Unsigned binary string → int."""
    return int(b, 2)
 
def sign_extend(value: int, bit_width: int) -> int:
    """Sign-extend a bit_width-wide value to a Python int."""
    sign_bit = 1 << (bit_width - 1)
    mask     = (1 << bit_width) - 1
    value   &= mask
    return (value ^ sign_bit) - sign_bit
 
 
# ─────────────────────────────────────────────
# Global CPU state
# ─────────────────────────────────────────────
 
pc                = 0          # current program counter
next_pc           = 0          # pc + 4 (computed in Fetch)
branch_target     = 0          # branch destination (computed in Execute)
alu_zero          = 0          # 1 when ALU result == 0
total_clock_cycles = 0         # incremented in Writeback
 
rf    = [0] * 32               # register file  (rf[0] always 0)
d_mem = [0] * 32               # data memory    (word-addressed: d_mem[i] = addr 4i)
 
# ── Control signals (set by ControlUnit, consumed by Execute/Mem/Writeback) ──
RegWrite = 0   # 1 → write result to rd
ALUSrc   = 0   # 0 → rs2,  1 → sign-extended immediate
MemRead  = 0   # 1 → load from data memory  (lw)
MemWrite = 0   # 1 → store to data memory   (sw)
MemToReg = 0   # 0 → ALU result to rd,  1 → memory data to rd
Branch   = 0   # 1 → instruction may branch (beq)
ALUOp    = 0   # 2-bit: 00=lw/sw, 01=beq, 10=R-type/I-type
 
# ── Decoded instruction fields (shared between pipeline "stages") ──
current_instr  = ""    # raw 32-bit binary string
opcode_bits    = ""
rs1_idx        = 0     # register file index
rs2_idx        = 0
rd_idx         = 0
rs1_val        = 0     # value read from rf[rs1_idx]
rs2_val        = 0     # value read from rf[rs2_idx]
imm_val        = 0     # sign-extended immediate
funct3         = ""
funct7         = ""
alu_ctrl       = 0b0000  # 4-bit ALU control code
alu_result     = 0     # output of ALU
mem_data       = 0     # data read from d_mem (lw)
mem_update_msg = ""
 
# RISC-V opcode constants
OP_R      = "0110011"
OP_IMM    = "0010011"
OP_LOAD   = "0000011"
OP_STORE  = "0100011"
OP_BRANCH = "1100011"
 
# RISC-V register ABI names (for display)
REG_NAMES = [
    "zero","ra","sp","gp","tp","t0","t1","t2",
    "s0","s1","a0","a1","a2","a3","a4","a5",
    "a6","a7","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","t3","t4","t5","t6"
]
 
# Storage for the full instruction list (loaded once in main)
instructions = []
 
 
# ─────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────
 
def Fetch():
    """
    Fetch the instruction at the current PC from the instruction list.
    PC is a byte address, so instruction index = pc // 4.
    Also computes next_pc = pc + 4.
    The actual PC update for the next cycle is performed later in Writeback(),
    after Execute has determined whether a branch is taken.
    """
    global pc, next_pc, current_instr
 
    instr_index = pc // 4
    if instr_index >= len(instructions):
        return False           # program finished
 
    current_instr = instructions[instr_index].strip()
    next_pc = pc + 4
 
    # # The branch mux lives conceptually in Fetch (it updates pc for next cycle).
    # # We apply it AFTER Execute has set branch_target and alu_zero.
    # # On the very first call Execute hasn't run yet so Branch=0 → safe.
    # if Branch and alu_zero:
    #     pc = branch_target
    # else:
    #     pc = next_pc
 
    return True
 
 
# ─────────────────────────────────────────────
# Control Unit
# ─────────────────────────────────────────────
 
def ControlUnit(opcode: str):
    """
    Receive 7-bit opcode string, set all control signals.
    Also derives alu_ctrl via inline ALU-control logic so
    Execute() receives a ready 4-bit code.
 
    ALU control table (from Processor-2 slide p.14):
      0000 = AND
      0001 = OR
      0010 = ADD
      0110 = SUB
      (beq uses SUB to compute rs1-rs2; zero flag signals equality)
    """
    global RegWrite, ALUSrc, MemRead, MemWrite, MemToReg, Branch, ALUOp
    global alu_ctrl
 
    # Reset all signals each cycle
    RegWrite = ALUSrc = MemRead = MemWrite = MemToReg = Branch = 0
    ALUOp    = 0b00
    alu_ctrl = 0b0010   # default ADD
 
    if opcode == OP_R:
        # R-type: add, sub, and, or
        RegWrite = 1
        ALUOp    = 0b10
        # alu_ctrl resolved below after funct3/funct7 are available
        # (set in Decode after ControlUnit returns)
 
    elif opcode == OP_IMM:
        # I-type ALU: addi, andi, ori
        RegWrite = 1
        ALUSrc   = 1
        ALUOp    = 0b10
 
    elif opcode == OP_LOAD:
        # lw
        RegWrite = 1
        ALUSrc   = 1
        MemRead  = 1
        MemToReg = 1
        ALUOp    = 0b00
        alu_ctrl = 0b0010   # ADD (address calculation)
 
    elif opcode == OP_STORE:
        # sw
        ALUSrc   = 1
        MemWrite = 1
        ALUOp    = 0b00
        alu_ctrl = 0b0010   # ADD
 
    elif opcode == OP_BRANCH:
        # beq
        Branch   = 1
        ALUOp    = 0b01
        alu_ctrl = 0b0110   # SUB (rs1 - rs2; zero → branch taken)
 
 
def resolve_alu_ctrl():
    """
    Called from Decode after funct3/funct7 are extracted.
    Only meaningful when ALUOp == 0b10 (R-type or I-type ALU).
    """
    global alu_ctrl
 
    if ALUOp != 0b10:
        return   # already set by ControlUnit for lw/sw/beq
 
    if opcode_bits == OP_R:
        if   funct3 == "000" and funct7 == "0000000":
            alu_ctrl = 0b0010   # ADD
        elif funct3 == "000" and funct7 == "0100000":
            alu_ctrl = 0b0110   # SUB
        elif funct3 == "111":
            alu_ctrl = 0b0000   # AND
        elif funct3 == "110":
            alu_ctrl = 0b0001   # OR
    else:
        # I-type (OP_IMM): addi / andi / ori
        if   funct3 == "000":
            alu_ctrl = 0b0010   # ADD  → addi
        elif funct3 == "111":
            alu_ctrl = 0b0000   # AND  → andi
        elif funct3 == "110":
            alu_ctrl = 0b0001   # OR   → ori
 
 
# ─────────────────────────────────────────────
# Decode
# ─────────────────────────────────────────────
 
def Decode():
    """
    Decode current_instr (32-bit binary string).
    Extracts register indices and reads their values from rf.
    Computes sign-extended immediate where applicable.
    Calls ControlUnit() then resolve_alu_ctrl().
    """
    global opcode_bits, rs1_idx, rs2_idx, rd_idx
    global rs1_val, rs2_val, imm_val, funct3, funct7
 
    instr = current_instr
    opcode_bits = slice_bits(instr, 6, 0)
 
    funct3 = slice_bits(instr, 14, 12)
    funct7 = slice_bits(instr, 31, 25)
 
    rd_idx  = bits_as_int(slice_bits(instr, 11, 7))
    rs1_idx = bits_as_int(slice_bits(instr, 19, 15))
    rs2_idx = bits_as_int(slice_bits(instr, 24, 20))
 
    # Read register values (rf[0] is always 0)
    rs1_val = rf[rs1_idx]
    rs2_val = rf[rs2_idx]
 
    # Immediate extraction (sign-extended)
    if opcode_bits in (OP_IMM, OP_LOAD):
        # I-type: imm[11:0] = bits 31:20
        raw = bits_as_int(slice_bits(instr, 31, 20))
        imm_val = sign_extend(raw, 12)
 
    elif opcode_bits == OP_STORE:
        # S-type: imm[11:5] = bits 31:25, imm[4:0] = bits 11:7
        hi  = slice_bits(instr, 31, 25)
        lo  = slice_bits(instr, 11, 7)
        raw = bits_as_int(hi + lo)
        imm_val = sign_extend(raw, 12)
 
    elif opcode_bits == OP_BRANCH:
        # SB-type immediate (from HW3 decode_sb)
        b31    = slice_bits(instr, 31, 31)
        b30_25 = slice_bits(instr, 30, 25)
        b11_8  = slice_bits(instr, 11, 8)
        b7     = slice_bits(instr, 7, 7)
        raw    = bits_as_int(b31 + b7 + b30_25 + b11_8)  # 12-bit branch immediate before implicit low-order 0
        imm_val = sign_extend(raw, 12)
 
    else:
        imm_val = 0   # R-type needs no immediate
 
    # Set control signals for this opcode
    ControlUnit(opcode_bits)
 
    # Resolve the 4-bit alu_ctrl code now that funct3/funct7 are known
    resolve_alu_ctrl()
 
 
# ─────────────────────────────────────────────
# Execute
# ─────────────────────────────────────────────
 
def Execute():
    """
    Run the ALU operation specified by alu_ctrl on the two operands.
    Operand B is either rs2_val (ALUSrc=0) or imm_val (ALUSrc=1).
 
    Also computes branch_target = (next_pc - 4) + (imm_val << 1).
    Note: next_pc was set to pc+4 in Fetch, so (next_pc - 4) == the
    PC of the current instruction, which is what the branch formula needs.
 
    Sets alu_zero = 1 if result == 0.
    """
    global alu_result, alu_zero, branch_target
 
    operand_a = rs1_val
    operand_b = imm_val if ALUSrc else rs2_val
 
    if   alu_ctrl == 0b0000:
        alu_result = operand_a & operand_b   # AND
    elif alu_ctrl == 0b0001:
        alu_result = operand_a | operand_b   # OR
    elif alu_ctrl == 0b0010:
        alu_result = operand_a + operand_b   # ADD
    elif alu_ctrl == 0b0110:
        alu_result = operand_a - operand_b   # SUB
    else:
        alu_result = 0
 
    # 1-bit zero flag
    alu_zero = 1 if alu_result == 0 else 0
 
    # Branch target = current PC + (branch immediate << 1)
    # imm_val does NOT include the implied low-order 0, so shift-left-1 restores byte offset.
    current_pc = next_pc - 4
    branch_target = current_pc + (imm_val << 1)

 
# ─────────────────────────────────────────────
# Mem
# ─────────────────────────────────────────────
 
def Mem():
    """
    Access data memory for lw / sw.
    d_mem is word-addressed: address 0x00 → d_mem[0],
                              address 0x04 → d_mem[1], etc.
    ALU result holds the byte address.
    """
    global mem_data, mem_update_msg
 
    mem_data = 0   # reset each cycle
    mem_update_msg = ""
 
    if MemRead:
        # lw: read word at alu_result
        mem_index = alu_result // 4
        mem_data  = d_mem[mem_index]
 
    elif MemWrite:
        # sw: write rs2_val to alu_result
        mem_index          = alu_result // 4
        d_mem[mem_index]   = rs2_val
        mem_update_msg = f"memory 0x{alu_result:X} is modified to 0x{rs2_val:X}"
 
 
# ─────────────────────────────────────────────
# Writeback
# ─────────────────────────────────────────────
 
def Writeback():
    """
    Write the result back to the destination register.
    MemToReg=1 → write mem_data (lw), MemToReg=0 → write alu_result.
    rf[0] is hardwired to 0 and never modified.
    Increments total_clock_cycles and prints cycle summary.
    """
    global total_clock_cycles, pc, next_pc, branch_target, Branch, alu_zero, mem_update_msg
 
    total_clock_cycles += 1
    print(f"total_clock_cycles {total_clock_cycles} :")
 
    if RegWrite and rd_idx != 0:
        write_val = mem_data if MemToReg else alu_result
        rf[rd_idx] = write_val
        reg_name   = f"x{rd_idx}"
        print(f"{reg_name} is modified to 0x{write_val & 0xFFFFFFFF:X}")
    
    if mem_update_msg:
        print(mem_update_msg)

    if Branch and alu_zero:
        pc = branch_target
    else:
        pc = next_pc

    print(f"pc is modified to 0x{pc:X}")
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
 
def main():
    global pc, rf, d_mem, branch_target, alu_zero, total_clock_cycles, mem_update_msg
 
    # ── Initial register values (as specified in the project) ──
    rf[1]  = 0x20   # x1
    rf[2]  = 0x5    # x2
    rf[10] = 0x70   # x10
    rf[11] = 0x4    # x11
 
    # ── Initial data memory values ──
    d_mem[0x70 // 4] = 0x5    # address 0x70 → d_mem[28]
    d_mem[0x74 // 4] = 0x10   # address 0x74 → d_mem[29]
 
    # ── Load program ──
    filename = input("Enter the program file name to run:\n").strip()
    try:
        with open(filename, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return
 
    global instructions
    instructions = lines
 
    # Reset PC and cycle counter
    pc                 = 0
    total_clock_cycles = 0
    branch_target      = 0
    alu_zero           = 0
 
    # ── Execution loop ──
    # Each iteration = one clock cycle (one instruction through all stages)
    while True:
        if not Fetch():
            break     # PC walked off the end of the program
        Decode()
        Execute()
        Mem()
        Writeback()
 
    print(f"\nprogram terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")
 
 
if __name__ == "__main__":
    main()