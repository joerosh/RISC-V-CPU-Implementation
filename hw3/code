# JoeSamuelRosh_NoahElliott.py
# RISC-V Instruction Decoder (CSE 140 Lab 3)
#
# Work Division for Demo:
#   Joe:
#     - Designed overall decoder architecture
#     - Implemented decode() dispatcher + per-type decode logic
#     - Implemented decode_sb() and decode_uj() immediate bit reconstruction and sign-extension
#     - Implemented shift-immediate handling (slli/srli/srai) in I-type decoder
#
#   Noah:
#     - Implemented bit extraction utilities (slice_bits, sign_extend, hex formatting)
#     - Implemented opcode/funct mapping tables
#     - Implemented CLI loop + formatted printing to match lab sample output
#
# Run:
#   python JoeSamuelRosh_NoahElliott.py


from dataclasses import dataclass
from typing import Dict, Any


# --------------
# isa_tables.py
# --------------

# Opcodes (7-bit)
OP_R = "0110011"
OP_IMM = "0010011"
OP_LOAD = "0000011"
OP_STORE = "0100011"
OP_BRANCH = "1100011"
OP_JALR = "1100111"
OP_JAL = "1101111"

# R-type: (funct3, funct7) -> op
R_TYPE_TABLE = {
    ("000", "0000000"): "add",
    ("000", "0100000"): "sub",
    ("111", "0000000"): "and",
    ("110", "0000000"): "or",
    ("100", "0000000"): "xor",
    ("001", "0000000"): "sll",
    ("101", "0000000"): "srl",
    ("101", "0100000"): "sra",
    ("010", "0000000"): "slt",
    ("011", "0000000"): "sltu",
}

# I-type OP-IMM: funct3 plus sometimes funct7 (for shifts)
I_IMM_SIMPLE = {
    "000": "addi",
    "010": "slti",
    "011": "sltiu",
    "100": "xori",
    "110": "ori",
    "111": "andi",
}

# LOAD: funct3 -> op
LOAD_TABLE = {
    "000": "lb",
    "001": "lh",
    "010": "lw",
}

# STORE: funct3 -> op
STORE_TABLE = {
    "000": "sb",
    "001": "sh",
    "010": "sw",
}

# BRANCH: funct3 -> op
BRANCH_TABLE = {
    "000": "beq",
    "001": "bne",
    "100": "blt",
    "101": "bge",
}


# -------------
# bit_utils.py 
# -------------

def bits_as_int(b: str) -> int:
    """Convert a binary string to int (unsigned)."""
    return int(b, 2)


def slice_bits(instr: str, hi: int, lo: int) -> str:
    """
    Return bits [hi:lo] inclusive as a string, where bit 31 is instr[0] and bit 0 is instr[31].
    Example: slice_bits(instr, 6, 0) returns the last 7 characters (opcode).
    """
    if len(instr) != 32:
        raise ValueError("Instruction must be 32 bits")
    start = 31 - hi
    end = 31 - lo
    return instr[start:end + 1]


def sign_extend(value: int, bit_width: int) -> int:
    """Sign-extend value interpreted as bit_width-bit signed integer to Python int."""
    sign_bit = 1 << (bit_width - 1)
    mask = (1 << bit_width) - 1
    value &= mask
    return (value ^ sign_bit) - sign_bit


def hex_masked(value: int, bit_width: int) -> str:
    """Return hex string like 0xFF0 masked to bit_width bits (uppercase A-F)."""
    mask = (1 << bit_width) - 1
    v = value & mask
    return "0x" + format(v, "X")


# -----------
# decoder.py
# -----------

@dataclass
class Decoded:
    inst_type: str
    operation: str
    fields: Dict[str, Any]


def reg_name(reg_bits: str) -> str:
    return f"x{bits_as_int(reg_bits)}"


def decode(instr: str) -> Decoded:
    instr = instr.strip()
    opcode = slice_bits(instr, 6, 0)

    if opcode == OP_R:
        return decode_r(instr)
    if opcode in (OP_IMM, OP_LOAD, OP_JALR):
        return decode_i(instr, opcode)
    if opcode == OP_STORE:
        return decode_s(instr)
    if opcode == OP_BRANCH:
        return decode_sb(instr)
    if opcode == OP_JAL:
        return decode_uj(instr)

    # Lab says inputs will be valid, so this should not happen.
    raise ValueError(f"Unsupported opcode: {opcode}")


def decode_r(instr: str) -> Decoded:
    rd = slice_bits(instr, 11, 7)
    funct3 = slice_bits(instr, 14, 12)
    rs1 = slice_bits(instr, 19, 15)
    rs2 = slice_bits(instr, 24, 20)
    funct7 = slice_bits(instr, 31, 25)

    op = R_TYPE_TABLE[(funct3, funct7)]

    fields = {
        "Rs1": reg_name(rs1),
        "Rs2": reg_name(rs2),
        "Rd": reg_name(rd),
        "Funct3": bits_as_int(funct3),
        "Funct7": bits_as_int(funct7),
    }
    return Decoded("R", op, fields)


def decode_i(instr: str, opcode: str) -> Decoded:
    rd = slice_bits(instr, 11, 7)
    funct3 = slice_bits(instr, 14, 12)
    rs1 = slice_bits(instr, 19, 15)
    imm12_bits = slice_bits(instr, 31, 20)
    imm_u = bits_as_int(imm12_bits)
    imm_s = sign_extend(imm_u, 12)

    # Determine operation based on opcode group
    if opcode == OP_LOAD:
        op = LOAD_TABLE[funct3]
        imm_val = imm_s
    elif opcode == OP_JALR:
        op = "jalr"
        imm_val = imm_s
    else:
        # OP-IMM includes shifts that need funct7 check (slli/srli/srai)
        if funct3 == "001":
            op = "slli"
            shamt = bits_as_int(slice_bits(instr, 24, 20))
            imm_val = shamt
        elif funct3 == "101":
            funct7 = slice_bits(instr, 31, 25)
            shamt = bits_as_int(slice_bits(instr, 24, 20))
            op = "srai" if funct7 == "0100000" else "srli"
            imm_val = shamt
        else:
            op = I_IMM_SIMPLE[funct3]
            imm_val = imm_s

    fields = {
        "Rs1": reg_name(rs1),
        "Rd": reg_name(rd),
        "Immediate": imm_val,
        "_imm_hex": hex_masked(imm_u, 12),
    }
    return Decoded("I", op, fields)


def decode_s(instr: str) -> Decoded:
    imm_hi = slice_bits(instr, 31, 25)   # imm[11:5]
    rs2 = slice_bits(instr, 24, 20)
    rs1 = slice_bits(instr, 19, 15)
    funct3 = slice_bits(instr, 14, 12)
    imm_lo = slice_bits(instr, 11, 7)    # imm[4:0]

    imm_bits = imm_hi + imm_lo
    imm_u = bits_as_int(imm_bits)
    imm_s = sign_extend(imm_u, 12)

    op = STORE_TABLE[funct3]
    fields = {
        "Rs1": reg_name(rs1),
        "Rs2": reg_name(rs2),
        "Immediate": imm_s,
        "_imm_hex": hex_masked(imm_u, 12),
    }
    return Decoded("S", op, fields)


def decode_sb(instr: str) -> Decoded:
    rs2 = slice_bits(instr, 24, 20)
    rs1 = slice_bits(instr, 19, 15)
    funct3 = slice_bits(instr, 14, 12)

    # SB immediate bits assembly:
    # imm[12] = bit31
    # imm[10:5] = bits30:25
    # imm[4:1] = bits11:8
    # imm[11] = bit7
    b31 = slice_bits(instr, 31, 31)
    b30_25 = slice_bits(instr, 30, 25)
    b11_8 = slice_bits(instr, 11, 8)
    b7 = slice_bits(instr, 7, 7)

    imm_bits = b31 + b7 + b30_25 + b11_8 + "0"  # 13 bits including bit0=0
    imm_u = bits_as_int(imm_bits)
    imm_s = sign_extend(imm_u, 13)

    op = BRANCH_TABLE[funct3]
    fields = {
        "Rs1": reg_name(rs1),
        "Rs2": reg_name(rs2),
        "Immediate": imm_s,
    }
    return Decoded("SB", op, fields)


def decode_uj(instr: str) -> Decoded:
    rd = slice_bits(instr, 11, 7)

    # UJ immediate bits assembly:
    # imm[20] = bit31
    # imm[10:1] = bits30:21
    # imm[11] = bit20
    # imm[19:12] = bits19:12
    b31 = slice_bits(instr, 31, 31)
    b30_21 = slice_bits(instr, 30, 21)
    b20 = slice_bits(instr, 20, 20)
    b19_12 = slice_bits(instr, 19, 12)

    imm_bits = b31 + b19_12 + b20 + b30_21 + "0"  # 21 bits including bit0=0
    imm_u = bits_as_int(imm_bits)
    imm_s = sign_extend(imm_u, 21)

    fields = {
        "Rd": reg_name(rd),
        "Immediate": imm_s,
        "_imm_hex": hex_masked(imm_u, 21),
    }
    return Decoded("UJ", "jal", fields)


# ---------
# main.py 
# ---------

def print_decoded(d: Decoded) -> None:
    print(f"Instruction Type: {d.inst_type}")
    print(f"Operation: {d.operation}")

    if d.inst_type == "R":
        print(f"Rs1: {d.fields['Rs1']}")
        print(f"Rs2: {d.fields['Rs2']}")
        print(f"Rd: {d.fields['Rd']}")
        print(f"Funct3: {d.fields['Funct3']}")
        print(f"Funct7: {d.fields['Funct7']}")
    elif d.inst_type == "I":
        print(f"Rs1: {d.fields['Rs1']}")
        print(f"Rd: {d.fields['Rd']}")
        imm = d.fields["Immediate"]
        imm_hex = d.fields["_imm_hex"]
        print(f"Immediate: {imm} (or {imm_hex})")
    elif d.inst_type == "S":
        print(f"Rs1: {d.fields['Rs1']}")
        print(f"Rs2: {d.fields['Rs2']}")
        imm = d.fields["Immediate"]
        imm_hex = d.fields["_imm_hex"]
        print(f"Immediate: {imm} (or {imm_hex})")
    elif d.inst_type == "SB":
        print(f"Rs1: {d.fields['Rs1']}")
        print(f"Rs2: {d.fields['Rs2']}")
        print(f"Immediate: {d.fields['Immediate']}")
    elif d.inst_type == "UJ":
        print(f"Rd: {d.fields['Rd']}")
        imm = d.fields["Immediate"]
        imm_hex = d.fields["_imm_hex"]
        print(f"Immediate: {imm} (or {imm_hex})")


def main() -> None:
    while True:
        print("Enter an instruction:")
        s = input().strip()
        if not s:
            break
        d = decode(s)
        print_decoded(d)
        print()


if __name__ == "__main__":
    main()